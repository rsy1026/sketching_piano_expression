#include<iostream>
#include<string>
#include<sstream>
#include<algorithm>
#include<cmath>
#include<vector>
#include<fstream>
#include<cassert>
#include"Hmm_v170225.hpp"
using namespace std;

int main(int argc, char** argv) {

	vector<int> v(100);
	vector<double> d(100);
	vector<string> s(100);
	stringstream ss;

	if(argc!=3){cout<<"Error in usage! : $./this in_fmt3x.txt out_hmm.txt"<<endl; return -1;}

	Fmt3x fmt3x;
	Hmm hmm;

	fmt3x.ReadFile(string(argv[1]));
	hmm.ConvertFromFmt3x(fmt3x);
	hmm.WriteFile(string(argv[2]));

	return 0;
}//end main
