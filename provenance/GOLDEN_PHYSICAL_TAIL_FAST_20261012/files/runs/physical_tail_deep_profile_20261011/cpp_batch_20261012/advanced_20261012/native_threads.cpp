// Isolated scalar NCC experiment. No production imports or changes.
// Compiled without vectorization, fast-math, contraction or reassociation.
#include <cstdint>
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

#include <immintrin.h>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <algorithm>


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

struct View {
    const uint8_t* data;
    int64_t rows, cols, row_stride, col_stride;
};
struct Task { View a,b; uint64_t support_id; };
struct Counters {
    uint64_t batches=0,tasks=0,conversions=0,centerings=0,norms=0,crossdots=0;
    uint64_t float_allocations=0,float_allocated_bytes=0,peak_float_bytes=0,support_hits=0;
};
struct Prepared { std::vector<double> x; double norm; };
struct Context {
    std::vector<double> scratch;
    std::vector<std::vector<double>> extra_scratch;
    std::unordered_map<uint64_t,Prepared> supports;
    Counters count;
    uint64_t live_support_bytes=0;
    explicit Context(uint64_t max_pixels):scratch(max_pixels) {
        for(int id=1;id<native_threads;++id) extra_scratch.emplace_back(max_pixels);



    }
};

// Sequential scalar reductions are deliberately exact-gated against NumPy.
// Identical formula and dtype do not imply identical BLAS reduction bits.
static double scalar_dot(const double* a,const double* b,int64_t n) {
    __m256d acc=_mm256_setzero_pd(); int64_t i=0;
    for(;i+4<=n;i+=4) acc=_mm256_add_pd(acc,_mm256_mul_pd(_mm256_loadu_pd(a+i),_mm256_loadu_pd(b+i)));
    double lanes[4]; _mm256_storeu_pd(lanes,acc);
    double result=((lanes[0]+lanes[1])+lanes[2])+lanes[3];
    for(;i<n;++i) result=result+a[i]*b[i];
    return result;
}
static double prepare(View v,double* x,Counters& count) {
    const int64_t n=v.rows*v.cols;
    double sum=0.0;
    int64_t k=0;
    for(int64_t r=0;r<v.rows;++r) for(int64_t c=0;c<v.cols;++c) {
        const double value=static_cast<double>(v.data[r*v.row_stride+c*v.col_stride]);
        x[k++]=value; sum=sum+value;
    }
    const double mean=sum/static_cast<double>(n);
    for(int64_t i=0;i<n;++i) x[i]=x[i]-mean;

    return scalar_dot(x,x,n);
}

extern "C" {
__declspec(dllexport) void* ncc_create(uint64_t max_pixels) {
    try { return new Context(max_pixels); } catch(...) { return nullptr; }
}
__declspec(dllexport) void ncc_destroy(void* handle) { delete static_cast<Context*>(handle); }
static int ncc_compute_batch(void* handle,const Task* tasks,uint64_t size,double* scores) {
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
__declspec(dllexport) void ncc_counters(void* handle,Counters* out) { *out=static_cast<Context*>(handle)->count; }
}
