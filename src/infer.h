#ifndef _INFERLDA_H
#define _INFERLDA_H

#include <string>

#include "pmat.h"
#include "pvec.h"

#include "doc.h"

using namespace std;

class Infer {
  private:
    int K;
    string type; // infer type
    bool verbose; // info printing from original code base

    string dfile;      // inference docs
    Pvec<double> pz;   // p(z) = theta
    Pmat<double> pw_z; // p(w|z) = phi, size K * M

  private:
    void load_para(string model_dir);

    void doc_infer(const Doc &doc, Pvec<double> &pz_d);
    void doc_infer_sum_b(const Doc &doc, Pvec<double> &pz_d);
    void doc_infer_sum_w(const Doc &doc, Pvec<double> &pz_d);
    void doc_infer_mix(const Doc &doc, Pvec<double> &pz_d);

    // compute condition distribution p(z|w, d) with p(w|z) fixed
    void compute_pz_dw(int w, const Pvec<double> &pz_d, Pvec<double> &p);

  public:
    Infer(string type, int K, bool verbose = true) : type(type), K(K), verbose(verbose) {}

    void run(string docs_pt, string output, string model_dir);
    void run_dynamic(string model_dir);
};

#endif

