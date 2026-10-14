// Isolated full proof port. Native math is the separately gated AVX2 batch.
// Python supplies the original set traversal permutation (not new candidates).
struct Span {int64_t a,b;};
struct VKey {
    View v;
    bool operator==(const VKey& b)const {return v.data==b.v.data&&v.rows==b.v.rows&&v.cols==b.v.cols&&v.row_stride==b.v.row_stride&&v.col_stride==b.v.col_stride;}
};
struct VHash {size_t operator()(const VKey& k)const {
    size_t r=reinterpret_cast<size_t>(k.v.data);
    for(auto n:{k.v.rows,k.v.cols,k.v.row_stride,k.v.col_stride})r^=static_cast<size_t>(n)+0x9e3779b9+(r<<6)+(r>>2);
    return r;
}};
struct PairKey {VKey a,b;bool operator==(const PairKey& k)const{return a==k.a&&b==k.b;}};
struct PairHash {size_t operator()(const PairKey& k)const{return VHash{}(k.a)^(VHash{}(k.b)<<1);}};
struct TraceRecord {View a,b;double score;};
struct ProofResult {const char* json;uint64_t bytes;const TraceRecord* trace;uint64_t requests,misses,supports,inner_batches;int code;};
static thread_local std::string proof_json;
static thread_local std::vector<TraceRecord> proof_trace;
static thread_local ProofResult proof_result;
static View crop(View v,int64_t lo,int64_t hi,int64_t width) {
    lo=std::max(int64_t(0),std::min(v.rows,lo));hi=std::max(lo,std::min(v.rows,hi));
    v.data+=lo*v.row_stride;v.rows=hi-lo;v.cols=width;return v;
}
static int64_t half_even(int64_t n) {
    int64_t q=n/2;if(n%2!=0&&q%2!=0)q+=(n>0?1:-1);return q;
}
struct RowResult {int64_t old_i,new_j;Span old_s,new_s;int64_t delta=0;double score=-1.;std::vector<std::pair<int64_t,uint64_t>> alignments;};
struct Candidate {int64_t shift,support;double boundary;uint64_t request;double score=-1.;std::vector<int64_t> entering;std::vector<RowResult> rows;};
struct Alternative {int64_t shift;uint64_t request;double score=-1.;};
class ProofCompute {
public:
    Context context;
    std::unordered_map<PairKey,uint64_t,PairHash> pairs;
    std::unordered_map<VKey,uint64_t,VHash> supports;
    std::vector<Task> tasks;
    std::vector<double> values;
    std::vector<uint64_t> trace_ids;
    uint64_t computed=0,batches=0;
    bool traced;
    explicit ProofCompute(uint64_t pixels,bool trace):context(pixels),traced(trace){}
    uint64_t request(View a,View b) {
        PairKey key{{a},{b}};auto found=pairs.find(key);uint64_t index;
        if(found!=pairs.end())index=found->second;
        else {
            auto support=supports.emplace(VKey{b},supports.size()+1).first;
            index=tasks.size();pairs.emplace(key,index);tasks.push_back(Task{a,b,support->second});values.push_back(-1.);
        }
        if(traced){proof_trace.push_back(TraceRecord{a,b,-1.});trace_ids.push_back(index);}
        return index;
    }
    void evaluate() {
        if(computed==tasks.size())return;
        const int code=ncc_compute_batch(&context,tasks.data()+computed,tasks.size()-computed,values.data()+computed);
        if(code)throw std::runtime_error("native batch failure");
        computed=tasks.size();++batches;
    }
};
// JSON numbers preserve float/int field types and binary64 round trips.
static void integer(std::string& s,int64_t v) {char b[32];auto r=std::to_chars(b,b+32,v);s.append(b,r.ptr);}
static void number(std::string& s,double v) {
    if(!std::isfinite(v))throw std::runtime_error("nonfinite proof score");
    char b[64];auto r=std::to_chars(b,b+64,v,std::chars_format::general,std::numeric_limits<double>::max_digits10);
    s.append(b,r.ptr);bool floating=false;for(char* p=b;p<r.ptr;++p)if(*p=='.'||*p=='e'||*p=='E')floating=true;
    if(!floating)s+=".0";
}
static void span_json(std::string& s,Span p) {s+='[';integer(s,p.a);s+=',';integer(s,p.b);s+=']';}
static void candidate_fields(std::string& s,const Candidate& c,bool best,const std::vector<Alternative>& alternatives,double margin,int64_t confident,double threshold) {
    s+="\"shift_px\":";integer(s,c.shift);s+=",\"score\":";number(s,c.score);
    s+=",\"boundary\":";number(s,c.boundary);s+=",\"support_end\":";integer(s,c.support);
    s+=",\"new_indices\":[";
    for(size_t j=0;j<c.entering.size();++j){if(j)s+=',';integer(s,c.entering[j]);}s+="],\"retained_rows\":[";
    for(size_t j=0;j<c.rows.size();++j) {
        if(j)s+=',';const auto& r=c.rows[j];s+="{\"old_index\":";integer(s,r.old_i);s+=",\"new_index\":";integer(s,r.new_j);
        s+=",\"old_span\":";span_json(s,r.old_s);s+=",\"new_span\":";span_json(s,r.new_s);
        s+=",\"row_alignment_residual_px\":";integer(s,r.delta);s+=",\"score\":";number(s,r.score);
        if(best){s+=",\"identity_status\":\"UNKNOWN\",\"pixel_anchor\":";s+=r.score>=threshold?"true":"false";}s+='}';
    }s+=']';
    if(best) {
        s+=",\"competing_retained_translations\":[";
        for(size_t i=0;i<std::min(size_t(3),alternatives.size());++i){if(i)s+=',';s+="{\"shift_px\":";integer(s,alternatives[i].shift);s+=",\"score\":";number(s,alternatives[i].score);s+='}';}
        s+="],\"margin\":";number(s,margin);s+=",\"confident_retained_rows\":";integer(s,confident);
    }
}
static void compute_proof(View old,View fresh,const Span* os,uint64_t on,const Span* ns,uint64_t nn,
    const int64_t* traversal,uint64_t traversal_n,double tol,double threshold,double margin_threshold,bool traced) {
    proof_json.clear();proof_trace.clear();proof_result={};
    if(old.rows!=fresh.rows||old.cols!=fresh.cols||on==0||nn==0) {
        proof_json="{\"status\":\"UNKNOWN\",\"reason\":\"NO_RETAINED_ROW_SUPPORT\",\"candidates\":[]}";return;
    }
    const int64_t height=fresh.rows,width=fresh.cols>=50?fresh.cols-12:fresh.cols;
    std::set<int64_t> shifts{0};
    for(uint64_t i=0;i<on;++i)for(uint64_t j=0;j<std::min(uint64_t(2),nn);++j) {
        const int64_t delta=half_even(os[i].a+os[i].b-ns[j].a-ns[j].b);
        for(int64_t d=std::max(int64_t(0),delta-1);d<std::min(height,delta+2);++d)shifts.insert(d);
    }
    // Reproduce OLD Python set iteration for alternative request order/ties.
    if(traversal_n!=shifts.size())throw std::runtime_error("shift traversal mismatch");
    std::set<int64_t> check(traversal,traversal+traversal_n);if(check!=shifts)throw std::runtime_error("shift set mismatch");
    ProofCompute p(old.rows*old.cols,traced);std::vector<Candidate> candidates;
    for(int64_t shift:shifts) {
        if(shift>0&&shift<=tol)continue;
        Candidate c;c.shift=shift;c.boundary=os[on-1].b-shift+tol;
        for(uint64_t j=0;j<nn;++j)if(ns[j].a>=c.boundary&&ns[j].b<height)c.entering.push_back(j);
        if(c.entering.empty())continue;
        std::set<int64_t> used;std::vector<std::pair<int64_t,int64_t>> retained;
        for(int64_t j=0;j<c.entering[0];++j) {
            int64_t selected=-1;int matches=0;
            for(uint64_t i=0;i<on;++i)if(!used.count(i)&&std::abs((ns[j].a+ns[j].b-os[i].a-os[i].b)/2.0+shift)<=tol){selected=i;++matches;}
            if(matches!=1){retained.clear();break;}
            used.insert(selected);retained.emplace_back(j,selected);
        }
        if(retained.size()<2)continue;
        bool valid=retained.size()==on-retained[0].second&&retained.size()==static_cast<size_t>(c.entering[0]);
        for(size_t k=0;k<retained.size();++k)if(retained[k].second!=retained[0].second+static_cast<int64_t>(k)||retained[k].first!=static_cast<int64_t>(k))valid=false;
        if(!valid)continue;
        c.support=std::min(height-shift,ns[c.entering[0]].a);
        if(shift==0) {
            bool occupied=false;
            for(int64_t j:c.entering) {
                int count=0;for(int64_t r=ns[j].a;r<ns[j].b;++r)for(int64_t col=0;col<width;++col)count+=old.data[r*old.row_stride+col*old.col_stride]>70;
                if(count>=5){occupied=true;break;}
            }if(occupied)continue;
        }
        c.request=p.request(crop(old,shift,shift+c.support,width),crop(fresh,0,c.support,width));
        for(auto [j,i]:retained) {
            RowResult r;r.old_i=i;r.new_j=j;r.old_s=os[i];r.new_s=ns[j];
            const int64_t lo=std::max(int64_t(0),ns[j].a-2),hi=std::min(c.support,ns[j].b+2);
            for(int64_t d=-static_cast<int64_t>(tol);d<=static_cast<int64_t>(tol);++d)if(lo+shift+d>=0&&hi+shift+d<=height)
                r.alignments.emplace_back(d,p.request(crop(old,lo+shift+d,hi+shift+d,width),crop(fresh,lo,hi,width)));
            if(r.alignments.empty())throw std::runtime_error("empty alignment list");
            c.rows.push_back(std::move(r));
        }
        candidates.push_back(std::move(c));
    }
    p.evaluate();
    for(auto& c:candidates) {
        c.score=p.values[c.request];
        for(auto& r:c.rows) {
            bool first=true;for(auto [d,k]:r.alignments)if(first||p.values[k]>r.score){first=false;r.delta=d;r.score=p.values[k];}
        }
    }
    std::sort(candidates.begin(),candidates.end(),[](const Candidate& a,const Candidate& b){return a.score>b.score||(a.score==b.score&&a.shift<b.shift);});
    std::vector<Alternative> alternatives;double margin=0.;int64_t confident=0;bool proven=false;
    const char* reason="NO_ORDERED_RETAINED_SUFFIX_AND_NEW_TAIL";
    if(!candidates.empty()) {
        auto& best=candidates[0];
        for(uint64_t i=0;i<traversal_n;++i) {
            int64_t shift=traversal[i];if(shift+best.support<=height&&std::abs(shift-best.shift)>tol)
                alternatives.push_back(Alternative{shift,p.request(crop(old,shift,shift+best.support,width),crop(fresh,0,best.support,width))});
        }
        p.evaluate();double second=-1.;bool first=true;
        for(auto& a:alternatives){a.score=p.values[a.request];if(first||a.score>second){second=a.score;first=false;}}
        std::stable_sort(alternatives.begin(),alternatives.end(),[](const Alternative& a,const Alternative& b){return a.score>b.score;});
        margin=best.score-second;
        for(auto& r:best.rows)confident+=r.score>=threshold;
        if(best.score<threshold||confident<2)reason="RETAINED_PIXELS_UNPROVEN";
        else if(margin<margin_threshold)reason="RETAINED_TRANSLATION_AMBIGUOUS";
        else {proven=true;reason="UNIQUE_PIXEL_SUFFIX_PREFIX_WITH_NEW_BOTTOM_REGION";}
    }
    proof_json="{\"status\":\"";proof_json+=proven?"PROVEN_PHYSICAL_NEW":"UNKNOWN";proof_json+="\",\"reason\":\"";proof_json+=reason;proof_json+="\",\"candidates\":[";
    for(size_t i=0;i<candidates.size();++i){if(i)proof_json+=',';proof_json+='{';candidate_fields(proof_json,candidates[i],i==0,alternatives,margin,confident,threshold);proof_json+='}';}
    proof_json+=']';if(proven){proof_json+=',';candidate_fields(proof_json,candidates[0],true,alternatives,margin,confident,threshold);}proof_json+='}';
    for(size_t i=0;i<p.trace_ids.size();++i)proof_trace[i].score=p.values[p.trace_ids[i]];
    proof_result.misses=p.tasks.size();proof_result.supports=p.supports.size();proof_result.inner_batches=p.batches;
}
extern "C" __declspec(dllexport) const ProofResult* ncc_proof_gil(View old,View fresh,const Span* os,uint64_t on,const Span* ns,uint64_t nn,
    const int64_t* traversal,uint64_t traversal_n,double tol,double threshold,double margin_threshold,int traced) {
    if(!save_thread||!restore_thread)return nullptr;
    auto before=Clock::now();void* state=save_thread();auto started=Clock::now();
    try {compute_proof(old,fresh,os,on,ns,nn,traversal,traversal_n,tol,threshold,margin_threshold,traced!=0);}
    catch(...) {proof_result={};proof_result.code=9;}
    proof_result.json=proof_json.data();proof_result.bytes=proof_json.size();proof_result.trace=proof_trace.data();proof_result.requests=proof_trace.size();
    auto acquire=Clock::now();restore_thread(state);auto ended=Clock::now();
    ++runtime.calls;runtime.native_wall_s+=seconds(started,acquire);runtime.gil_release_s+=seconds(before,started);runtime.gil_restore_s+=seconds(acquire,ended);
    return &proof_result;
}
