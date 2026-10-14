"""Generate separate AVX2 batch with persistent native compute pool + GIL timing."""
from pathlib import Path
p=Path(__file__).resolve().parent
s=(p.parent/'avx2_uncounted.cpp').read_text()
s=s.replace('#include <cstdint>','''#include <cstdint>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <memory>
#include <atomic>
#include <chrono>
#include <string>
#include <charconv>
#include <limits>
#include <set>
#include <cstring>
#include <stdexcept>
''')
pool='''
static int fail_worker_start=-1; // isolated resource-failure injection only
class NativePool {
    std::vector<std::thread> workers;
    std::mutex mutex; std::condition_variable launch,done;
    bool stopping=false; uint64_t generation=0; int remaining=0;
    std::function<void(int)> work;
public:
    const int count;
    explicit NativePool(int n):count(n) {
        try { for(int id=1;id<n;++id) {
        if(id==fail_worker_start)throw std::runtime_error("injected native worker start failure");
        workers.emplace_back([this,id] {
            uint64_t seen=0;
            for(;;) {
                std::unique_lock<std::mutex> lock(mutex);
                launch.wait(lock,[&]{return stopping||generation!=seen;});
                if(stopping)return;
                seen=generation; auto* job=&work; lock.unlock(); (*job)(id); lock.lock();
                --remaining; if(!remaining)done.notify_one();
            }
        });
        }} catch(...) {
            {std::lock_guard<std::mutex> lock(mutex);stopping=true;}
            launch.notify_all();for(auto& t:workers)if(t.joinable())t.join();throw;
        }
    }
    ~NativePool() {
        {std::lock_guard<std::mutex> lock(mutex);stopping=true;}
        launch.notify_all();for(auto& t:workers)t.join();
    }
    void run(std::function<void(int)> job) {
        if(count==1){job(0);return;}
        {std::lock_guard<std::mutex> lock(mutex);work=job;remaining=count-1;++generation;}
        launch.notify_all(); job(0);
        std::unique_lock<std::mutex> lock(mutex);done.wait(lock,[&]{return remaining==0;});
        work={};
    }
};
static std::unique_ptr<NativePool> pool;
static int native_threads=1;
using SaveThread=void* (*)(); using RestoreThread=void (*)(void*);
static SaveThread save_thread=nullptr; static RestoreThread restore_thread=nullptr;
using Clock=std::chrono::steady_clock;
static double seconds(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double>(b-a).count();}
struct Runtime {uint64_t calls=0;double native_wall_s=0,gil_release_s=0,gil_restore_s=0;};
static thread_local Runtime runtime;
'''
s=s.replace('struct View {',pool+'\nstruct View {')
s=s.replace('    std::vector<double> scratch;','    std::vector<double> scratch;\n    std::vector<std::vector<double>> extra_scratch;')
s=s.replace('explicit Context(uint64_t max_pixels):scratch(max_pixels) {','explicit Context(uint64_t max_pixels):scratch(max_pixels) {\n        for(int id=1;id<native_threads;++id) extra_scratch.emplace_back(max_pixels);')
start=s.index('__declspec(dllexport) int ncc_batch(')
end=s.index('__declspec(dllexport) void ncc_counters',start)
s=s[:start]+'''static int ncc_compute_batch(void* handle,const Task* tasks,uint64_t size,double* scores) {
    Context& ctx=*static_cast<Context*>(handle);
    std::vector<const Prepared*> prepared(size,nullptr);
    // All shared cache mutation finishes before native compute workers start.
    for(uint64_t j=0;j<size;++j) {
        const Task& task=tasks[j]; const int64_t n=task.a.rows*task.a.cols;
        if(task.a.rows!=task.b.rows||task.a.cols!=task.b.cols||n==0){scores[j]=-1.0;continue;}
        if(static_cast<uint64_t>(n)>ctx.scratch.size())return 2;
        auto found=ctx.supports.find(task.support_id);
        if(found==ctx.supports.end()) {
            Prepared p;p.x.resize(n);p.norm=prepare(task.b,p.x.data(),ctx.count);
            found=ctx.supports.emplace(task.support_id,std::move(p)).first;
        }
        prepared[j]=&found->second;
    }
    std::atomic<uint64_t> next{0};
    auto compute=[&](int id){
        double* scratch=id==0?ctx.scratch.data():ctx.extra_scratch[id-1].data();
        Counters local_unused;
        for(;;) {
            uint64_t j=next.fetch_add(1,std::memory_order_relaxed);if(j>=size)break;
            if(!prepared[j])continue;
            const Task& task=tasks[j];const int64_t n=task.a.rows*task.a.cols;
            const double xx=prepare(task.a,scratch,local_unused),yy=prepared[j]->norm;
            const double den=std::sqrt(xx*yy);
            scores[j]=den!=0.0?scalar_dot(scratch,prepared[j]->x.data(),n)/den:-1.0;
        }
    };
    if(pool)pool->run(compute);else compute(0);
    return 0;
}
__declspec(dllexport) int ncc_config_threads(int n) {
    if(n<1||n>4)return 2;
    try {
        auto next=n>1?std::make_unique<NativePool>(n):nullptr;
        pool=std::move(next);native_threads=n;return 0;
    } catch(...) {return 9;}
}
__declspec(dllexport) void ncc_test_fail_thread(int id){fail_worker_start=id;}
__declspec(dllexport) void ncc_bind_gil(void* save,void* restore) {
    save_thread=reinterpret_cast<SaveThread>(save);restore_thread=reinterpret_cast<RestoreThread>(restore);
}
__declspec(dllexport) void ncc_runtime_reset(){runtime={};}
__declspec(dllexport) void ncc_runtime_get(Runtime* out){*out=runtime;}
__declspec(dllexport) int ncc_batch(void* handle,const Task* tasks,uint64_t size,double* scores) {
    if(!save_thread||!restore_thread)return 7;
    auto before=Clock::now();void* state=save_thread();auto started=Clock::now();
    int code=0;try{code=ncc_compute_batch(handle,tasks,size,scores);}catch(...){code=9;}
    auto acquire=Clock::now();restore_thread(state);auto ended=Clock::now();
    ++runtime.calls;runtime.native_wall_s+=seconds(started,acquire);
    runtime.gil_release_s+=seconds(before,started);runtime.gil_restore_s+=seconds(acquire,ended);
    return code;
}
__declspec(dllexport) void ncc_gil_pause(int milliseconds) {
    void* state=save_thread();std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));restore_thread(state);
}
''' +s[end:]
(p/'native_threads.cpp').write_text(s)
