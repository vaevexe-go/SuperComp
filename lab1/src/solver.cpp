// Task 1, variant 15: serial and OpenMP solvers.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
using std::fmax;using std::fmin;using std::fabs;using std::sqrt;using std::asin;using std::pow;

/* Integral of sqrt(9-x*x). Break at all intersections with cell edges.
   On each resulting interval the clipped vertical length is a+b*sqrt(9-x*x).
   Thus the circle/cell intersection area is integrated analytically. */
static double primitive(double x) {
    x=fmax(-3.0,fmin(3.0,x));
    return 0.5*(x*sqrt(fmax(0.0,9-x*x))+9*asin(x/3));
}
static void add_root(double *cuts,int *n,double s,double a,double b) {
    if(s<0 || s>3) return;
    double r=sqrt(fmax(0.0,9-s*s));
    if(-r>a && -r<b) cuts[(*n)++]=-r;
    if(r>a && r<b) cuts[(*n)++]=r;
}
static double area(double x0,double x1,double y0,double y1) {
    double xmin=(x0<=0 && x1>=0)?0:fmin(fabs(x0),fabs(x1));
    double xmax=fmax(fabs(x0),fabs(x1));
    double ymin1=(y0<=2 && y1>=2)?0:fmin(fabs(y0-2),fabs(y1-2));
    double ymax1=fmax(fabs(y0-2),fabs(y1-2));
    double ymin2=(y0<=-2 && y1>=-2)?0:fmin(fabs(y0+2),fabs(y1+2));
    double ymax2=fmax(fabs(y0+2),fabs(y1+2));
    if(xmax*xmax+ymax1*ymax1<=9 && xmin*xmin+ymin2*ymin2>=9)
        return (x1-x0)*(y1-y0);
    if(xmin*xmin+ymin1*ymin1>=9 || xmax*xmax+ymax2*ymax2<=9)
        return 0;
    double a=fmax(-3,x0),b=fmin(3,x1),cuts[32]; int n=0;
    if(a>=b) return 0;
    cuts[n++]=a; cuts[n++]=b;
    if(a<0 && b>0) cuts[n++]=0;
    add_root(cuts,&n,2,a,b);
    for(int k=0;k<2;k++) {
        double y=k?y1:y0;
        add_root(cuts,&n,y-2,a,b);  /* top: 2+s=y */
        add_root(cuts,&n,2-y,a,b);  /* lower: 2-s=y */
        add_root(cuts,&n,y+2,a,b);  /* lower: s-2=y */
    }
    for(int i=1;i<n;i++) { double z=cuts[i]; int j=i; while(j>0 && cuts[j-1]>z){cuts[j]=cuts[j-1];j--;}cuts[j]=z; }
    double result=0;
    for(int i=0;i<n-1;i++) {
        double l=cuts[i],r=cuts[i+1],s=sqrt(fmax(0,9-pow((l+r)*0.5,2)));
        double hi=fmin(y1,2+s),lo=fmax(y0,fabs(s-2));
        if(hi<=lo || r<=l) continue;
        double c=y1,d=0;
        if(2+s<y1){c=2;d=1;}
        if(fabs(s-2)>y0){if(s>=2){c+=2;d-=1;}else{c-=2;d+=1;}}
        else c-=y0;
        result+=c*(r-l)+d*(primitive(r)-primitive(l));
    }
    return fmax(0,fmin((x1-x0)*(y1-y0),result));
}

struct Grid {
    int M, N;
    double hx, hy;
    Grid(int m, int n) : M(m), N(n), hx(6.0/m), hy(5.0/n) {}
    std::size_t index(int i, int j) const {
        return static_cast<std::size_t>(i)*(N+1)+j;
    }
    std::size_t size() const { return static_cast<std::size_t>(M+1)*(N+1); }
};
using Vector = std::vector<double>;

static double dot(const Grid& g, const Vector& a, const Vector& b) {
    double sum=0;
    #pragma omp parallel for collapse(2) reduction(+:sum) schedule(static)
    for(int i=1;i<g.M;++i) for(int j=1;j<g.N;++j) {
        const std::size_t k=g.index(i,j);
        sum+=a[k]*b[k];
    }
    return sum*g.hx*g.hy;
}

static void apply(const Grid& g, const Vector& v, const Vector& diag, Vector& out) {
    const double ax=1/(g.hx*g.hx), ay=1/(g.hy*g.hy);
    #pragma omp parallel for collapse(2) schedule(static)
    for(int i=1;i<g.M;++i) for(int j=1;j<g.N;++j) {
        const std::size_t k=g.index(i,j);
        out[k]=diag[k]*v[k]-ax*(v[g.index(i-1,j)]+v[g.index(i+1,j)])
                            -ay*(v[k-1]+v[k+1]);
    }
}

static bool in_domain(double x, double y) {
    return x*x+(y-2)*(y-2)<9 && x*x+(y+2)*(y+2)>9;
}

static void save_solution(const Grid& g, const Vector& x, const std::string& name) {
    std::ofstream out(name.c_str());
    if(!out) throw std::runtime_error("Cannot write "+name);
    out << std::setprecision(17) << "i,j,x,y,u,in_domain\n";
    for(int i=0;i<=g.M;++i) for(int j=0;j<=g.N;++j) {
        const double xx=-3+i*g.hx, yy=j*g.hy;
        const bool boundary=i==0||j==0||i==g.M||j==g.N;
        out << i << ',' << j << ',' << xx << ',' << yy << ','
            << (boundary?std::sin(xx+yy):x[g.index(i,j)]) << ','
            << in_domain(xx,yy) << '\n';
    }
    if(!out) throw std::runtime_error("Write failed: "+name);
}

static Vector read_reference(const Grid& g, const std::string& name) {
    std::ifstream in(name.c_str());
    if(!in) throw std::runtime_error("Cannot read reference "+name);
    std::string line;std::getline(in,line);
    if(line!="i,j,x,y,u,in_domain") throw std::runtime_error("Invalid reference header");
    Vector result(g.size(),0.0);
    std::size_t count=0;
    while(std::getline(in,line)) {
        std::replace(line.begin(),line.end(),',',' ');
        std::istringstream row(line);
        int i,j,inside;double xx,yy,u;
        if(!(row>>i>>j>>xx>>yy>>u>>inside) || i<0||i>g.M||j<0||j>g.N ||
           g.index(i,j)!=count || std::abs(xx-(-3+i*g.hx))>1e-10 ||
           std::abs(yy-j*g.hy)>1e-10 || !std::isfinite(u))
            throw std::runtime_error("Reference grid does not match");
        result[count++]=u;
    }
    if(count!=g.size()) throw std::runtime_error("Incomplete reference grid");
    return result;
}

static void write_history(std::ostream& out,const Grid& g,int iteration,
                          double step,double functional,const Vector& x,
                          const Vector& b,const Vector& diag,Vector& ax,
                          const Vector& reference,bool manufactured,double bnorm) {
    apply(g,x,diag,ax);
    double rr=0,ref2=0,refmax=0,exact2=0,exactmax=0;
    for(int i=1;i<g.M;++i) for(int j=1;j<g.N;++j) {
        const std::size_t k=g.index(i,j);
        const double r=b[k]-ax[k];rr+=r*r;
        if(!reference.empty()) {
            const double e=x[k]-reference[k];ref2+=e*e;
            refmax=std::max(refmax,std::abs(e));
        }
        if(manufactured) {
            const double e=x[k]-std::sin(-3+i*g.hx+j*g.hy);exact2+=e*e;
            exactmax=std::max(exactmax,std::abs(e));
        }
    }
    const double nan=std::numeric_limits<double>::quiet_NaN();
    const double residual=std::sqrt(rr*g.hx*g.hy);
    out<<iteration<<','<<step<<','<<functional<<','<<residual<<','<<residual/bnorm<<','
       <<(reference.empty()?nan:std::sqrt(ref2*g.hx*g.hy))<<','
       <<(reference.empty()?nan:refmax)<<','
       <<(manufactured?std::sqrt(exact2*g.hx*g.hy):nan)<<','
       <<(manufactured?exactmax:nan)<<'\n';
}

static int positive_int(const std::string& s) {
    std::size_t end=0;const long n=std::stol(s,&end);
    if(end!=s.size()||n<1||n>std::numeric_limits<int>::max()-1)
        throw std::invalid_argument("Invalid positive integer: "+s);
    return static_cast<int>(n);
}

static int run(int argc,char** argv) {
    if(argc<3) {
        std::cerr<<"Usage: solver M N [delta=1e-9] [solution.csv|-] [history.csv|-] [manufactured=0]"
                 <<" [--reference solution.csv] [--snapshots directory] [--snapshot-every 50]\n";
        return 2;
    }
    const int M=positive_int(argv[1]),N=positive_int(argv[2]);
    std::size_t end=0;
    const std::string tolerance=argc>3?argv[3]:"1e-9";
    const double delta=std::stod(tolerance,&end);
    if(M<3||N<3||end!=tolerance.size()||!std::isfinite(delta)||delta<=0)
        throw std::invalid_argument("Require M,N >= 3 and finite delta > 0");
    const std::string solution=argc>4?argv[4]:"-",history=argc>5?argv[5]:"-";
    const std::string manufactured_arg=argc>6?argv[6]:"0";
    if(manufactured_arg!="0"&&manufactured_arg!="1")
        throw std::invalid_argument("manufactured must be 0 or 1");
    const bool manufactured=manufactured_arg=="1";
    std::string reference_name,snapshot_dir;int snapshot_every=50;
    for(int a=7;a<argc;a+=2) {
        if(a+1==argc) throw std::invalid_argument("Missing option value");
        const std::string option=argv[a];
        if(option=="--reference") reference_name=argv[a+1];
        else if(option=="--snapshots") snapshot_dir=argv[a+1];
        else if(option=="--snapshot-every") snapshot_every=positive_int(argv[a+1]);
        else throw std::invalid_argument("Unknown option: "+option);
    }
    const Grid g(M,N);
    if(g.size()>std::numeric_limits<std::size_t>::max()/(8*sizeof(double)))
        throw std::length_error("Grid too large");
    Vector x(g.size(),0),r(g.size(),0),z(g.size(),0),p(g.size(),0),ap(g.size(),0),b(g.size(),0),diag(g.size(),0);
    const double hx=g.hx,hy=g.hy,h=std::max(hx,hy),eps=h*h,ax=1/(hx*hx),ay=1/(hy*hy);
    #pragma omp parallel for collapse(2) schedule(static)
    for(int i=1;i<M;++i) for(int j=1;j<N;++j) {
        const std::size_t k=g.index(i,j);const double xx=-3+i*hx,yy=j*hy;
        double fraction=area(xx-hx/2,xx+hx/2,yy-hy/2,yy+hy/2)/(hx*hy);
        fraction=std::max(0.0,std::min(1.0,fraction));
        const double q=(1-fraction)/(eps*eps),v=std::sin(xx+yy);
        diag[k]=2*ax+2*ay+q;b[k]=(manufactured?2*v:fraction)+q*v;
        if(i==1)b[k]+=ax*std::sin(-3+yy);
        if(i==M-1)b[k]+=ax*std::sin(3+yy);
        if(j==1)b[k]+=ay*std::sin(xx);
        if(j==N-1)b[k]+=ay*std::sin(xx+5);
        r[k]=b[k];z[k]=r[k]/diag[k];p[k]=z[k];
    }
    int threads=1;
#ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp single
        threads=omp_get_num_threads();
    }
#endif
    const Vector reference=reference_name.empty()?Vector():read_reference(g,reference_name);
    std::ofstream trace;
    if(history!="-") {
        trace.open(history.c_str());if(!trace)throw std::runtime_error("Cannot write "+history);
        trace<<std::setprecision(17)
             <<"iteration,step_norm,functional,residual_norm,relative_residual,iter_error_l2,iter_error_max,exact_error_l2,exact_error_max\n";
    }
    const double bnorm=std::sqrt(dot(g,b,b));double rho=dot(g,r,z);
    const double nan=std::numeric_limits<double>::quiet_NaN();
    double step=std::numeric_limits<double>::infinity(),Jold=0;
    int iteration=0,restarts=0;bool converged=false;
    const long long limit=static_cast<long long>(M-1)*(N-1);
    auto snapshot=[&](int k) {
        if(!snapshot_dir.empty()) {
            std::ostringstream name;name<<snapshot_dir<<"/snapshot_"<<std::setw(5)<<std::setfill('0')<<k<<".csv";
            save_solution(g,x,name.str());
        }
    };
    if(trace.is_open())write_history(trace,g,0,nan,0,x,b,diag,ap,reference,manufactured,bnorm);
    snapshot(0);
    const auto start=std::chrono::steady_clock::now();
    while(iteration<limit) {
        apply(g,p,diag,ap);const double den=dot(g,p,ap);
        if(!(den>0)||!(rho>0)||!std::isfinite(den)) throw std::runtime_error("PCG breakdown");
        const double alpha=rho/den;double s2=0,rz=0,Jsum=0;
        #pragma omp parallel reduction(+:s2,rz,Jsum)
        {
            // Compensated sums keep the penalty-dominated energy accurate.
            double compensation=0;
            #pragma omp for collapse(2) schedule(static)
            for(int i=1;i<M;++i) for(int j=1;j<N;++j) {
                const std::size_t k=g.index(i,j);const double d=alpha*p[k];
                x[k]+=d;r[k]-=alpha*ap[k];z[k]=r[k]/diag[k];
                s2+=d*d;rz+=r[k]*z[k];
                const double term=(b[k]+r[k])*x[k]-compensation,total=Jsum+term;
                compensation=(total-Jsum)-term;Jsum=total;
            }
        }
        step=std::sqrt(s2*hx*hy);const double next=rz*hx*hy,Jnew=-0.5*Jsum*hx*hy;++iteration;
        if(Jnew>Jold+256*std::numeric_limits<double>::epsilon()*std::max(1.0,std::abs(Jold))) {
            #pragma omp parallel for collapse(2) schedule(static)
            for(int i=1;i<M;++i) for(int j=1;j<N;++j) { const auto k=g.index(i,j);x[k]-=alpha*p[k]; }
            apply(g,x,diag,ap);
            #pragma omp parallel for collapse(2) schedule(static)
            for(int i=1;i<M;++i) for(int j=1;j<N;++j) {const auto k=g.index(i,j);r[k]=b[k]-ap[k];z[k]=r[k]/diag[k];p[k]=z[k];}
            rho=dot(g,r,z);
            if(++restarts>20)throw std::runtime_error("Too many energy restarts");
            if(trace.is_open())write_history(trace,g,iteration,step,Jold,x,b,diag,ap,reference,manufactured,bnorm);
            if(iteration%snapshot_every==0)snapshot(iteration);
            continue;
        }
        Jold=Jnew;
        if(trace.is_open())write_history(trace,g,iteration,step,Jold,x,b,diag,ap,reference,manufactured,bnorm);
        if(iteration%snapshot_every==0)snapshot(iteration);
        if(step<delta) {
            apply(g,x,diag,ap);
            #pragma omp parallel for collapse(2) schedule(static)
            for(int i=1;i<M;++i) for(int j=1;j<N;++j) {const auto k=g.index(i,j);r[k]=b[k]-ap[k];z[k]=r[k]/diag[k];}
            if(std::sqrt(dot(g,r,r))/bnorm<1e-8){converged=true;break;}
            #pragma omp parallel for collapse(2) schedule(static)
            for(int i=1;i<M;++i) for(int j=1;j<N;++j) {const auto k=g.index(i,j);p[k]=z[k];}
            rho=dot(g,r,z);++restarts;continue;
        }
        const double beta=next/rho;
        #pragma omp parallel for collapse(2) schedule(static)
        for(int i=1;i<M;++i) for(int j=1;j<N;++j) {const auto k=g.index(i,j);p[k]=z[k]+beta*p[k];}
        rho=next;
    }
    const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    snapshot(iteration);
    if(trace.is_open()&&!trace)throw std::runtime_error("History write failed");
    apply(g,x,diag,ap);
    double err2=0,errmax=0,norm2=0,rs2=0,Jtrue=0;
    #pragma omp parallel for collapse(2) reduction(+:err2,norm2,rs2,Jtrue) reduction(max:errmax) schedule(static)
    for(int i=1;i<M;++i) for(int j=1;j<N;++j) {
        const auto k=g.index(i,j);const double e=x[k]-std::sin(-3+i*hx+j*hy),rr=b[k]-ap[k];
        err2+=e*e;errmax=std::max(errmax,std::abs(e));norm2+=x[k]*x[k];rs2+=rr*rr;Jtrue+=x[k]*(0.5*ap[k]-b[k]);
    }
#ifdef _OPENMP
    const char* mode="openmp";
#else
    const char* mode="serial";
#endif
    std::cout<<std::setprecision(12)<<mode<<','<<M<<','<<N<<','<<threads<<','<<iteration<<','<<elapsed<<','
             <<step<<','<<std::sqrt(rs2*hx*hy)/bnorm<<','<<std::sqrt(norm2*hx*hy)<<','<<Jtrue*hx*hy<<','
             <<restarts<<','<<converged<<','<<(manufactured?std::sqrt(err2*hx*hy):nan)<<','
             <<(manufactured?errmax:nan)<<','<<eps<<','<<manufactured<<'\n';
    if(solution!="-")save_solution(g,x,solution);
    return converged?0:1;
}

int main(int argc,char** argv) {
    try { return run(argc,argv); }
    catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 2; }
}
