#define main solver_program_main
#include "../src/solver.cpp"
#undef main
#include <cstdio>
#include <random>
#include <unistd.h>

static void require(bool condition,const std::string& message) {
    if(!condition)throw std::runtime_error(message);
}
static std::string solve(int m,int n,double delta,bool exact,int threads,
                         const std::string& solution,const std::string& history="-") {
#ifdef _OPENMP
    omp_set_dynamic(0);omp_set_num_threads(threads);
#else
    (void)threads;
#endif
    std::ostringstream tol;tol<<std::setprecision(17)<<delta;
    std::vector<std::string> args={"solver",std::to_string(m),std::to_string(n),tol.str(),solution,history,exact?"1":"0"};
    std::vector<char*> argv;for(auto& arg:args)argv.push_back(&arg[0]);
    std::ostringstream output;auto old=std::cout.rdbuf(output.rdbuf());int status;
    try {status=run(static_cast<int>(argv.size()),argv.data());}
    catch(...) {std::cout.rdbuf(old);throw;}
    std::cout.rdbuf(old);require(status==0,"Solver did not converge");return output.str();
}
static double exact_error(const Grid& g,const Vector& u) {
    double error=0;
    for(int i=1;i<g.M;++i)for(int j=1;j<g.N;++j)
        error=std::max(error,std::abs(u[g.index(i,j)]-std::sin(-3+i*g.hx+j*g.hy)));
    return error;
}
int main() {
    std::vector<std::string> files;std::string temp;
    try {
        char pattern[]="/tmp/lab15-test-XXXXXX";char* dir=mkdtemp(pattern);
        require(dir!=nullptr,"Cannot create temporary directory");temp=dir;
        auto file=[&](const std::string& name){files.push_back(temp+"/"+name);return files.back();};
        const double pi=std::acos(-1.0);
        const double expected=9*pi-18*std::acos(2.0/3)+4*std::sqrt(5.0);
        require(std::abs(area(-3,3,0,5)-expected)<1e-11,"Total domain area");
        require(area(-0.1,0.1,2,2.2)>0.03999999999,"Interior cell area");
        require(area(-0.1,0.1,0,0.1)==0,"Exterior cell area");
        std::mt19937 random(60915);std::uniform_real_distribution<double> ux(-3.1,3.0),uy(-0.1,4.9),width(0.001,0.5);
        for(int k=0;k<300;++k) {
            const double x=ux(random),y=uy(random),w=width(random),h=width(random),s=area(x,x+w,y,y+h);
            require(s>=0&&s<=w*h+1e-14,"Area bounds");
            const double split=area(x,x+w/2,y,y+h)+area(x+w/2,x+w,y,y+h);
            require(std::abs(split-s)<1e-10,"Area additivity");
        }
        std::cout<<"PASS: analytic area, interior/exterior cells, 300 bounds/additivity checks\n";
        double previous=0;
        for(int n:{20,40,80}) {
            const auto name=file("exact_"+std::to_string(n)+".csv");solve(n,n,1e-11,true,1,name);
            const double error=exact_error(Grid(n,n),read_reference(Grid(n,n),name));
            if(previous>0)require(previous/error>3.5&&previous/error<4.5,"Manufactured second order");
            previous=error;
        }
        std::cout<<"PASS: manufactured solution on 20, 40, 80; second order\n";
        const auto reference_name=file("serial.csv"),trace_name=file("trace.csv");
        const auto metrics=solve(80,80,1e-9,false,1,reference_name,trace_name);
        const Grid g(80,80);const auto reference=read_reference(g,reference_name);
        std::ifstream trace(trace_name.c_str());std::string line;int records=-1;
        double final_step=1,last_functional=0;
        while(std::getline(trace,line)) {
            if(++records==0)continue;
            std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);
            int it;double step,functional;row>>it;
            // The first row deliberately has an undefined step (nan).
            if(it==0)continue;
            require(bool(row>>step>>functional),"Malformed history row");
            require(functional<=last_functional+1e-7,"Functional monotonicity");
            last_functional=functional;final_step=step;
        }
        require(records==262,"One history row for each iteration plus initial state");
        require(final_step<1e-9,"Final step tolerance");
#ifdef _OPENMP
        for(int threads:{4,16}) {
            const auto name=file("parallel_"+std::to_string(threads)+".csv");solve(80,80,1e-9,false,threads,name);
            const auto u=read_reference(g,name);double error=0;
            for(std::size_t k=0;k<u.size();++k)error=std::max(error,std::abs(u[k]-reference[k]));
            require(error<1e-10,"Thread consistency");
        }
        std::cout<<"PASS: consistent solutions on 1, 4, 16 threads\n";
#endif
        std::cout<<"PASS: true-residual convergence, iteration history, monotone functional\n";
        for(const auto& path:files)std::remove(path.c_str());rmdir(temp.c_str());
        return 0;
    } catch(const std::exception& e) {
        for(const auto& path:files)std::remove(path.c_str());if(!temp.empty())rmdir(temp.c_str());
        std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;
    }
}
