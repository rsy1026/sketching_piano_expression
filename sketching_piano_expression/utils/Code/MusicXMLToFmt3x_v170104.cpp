#include<iostream>
#include<string>
#include<sstream>
#include<cmath>
#include<vector>
#include<algorithm>
#include<fstream>
#include<cassert>
#include"Fmt3x_v170225.hpp"
using namespace std;

int main(int argc, char** argv) {

	vector<int> v(100);
	vector<double> d(100);
	vector<string> s(100);
	stringstream ss;

	if(argc!=3){cout<<"Error in usage! : $./this in.xml out_fmt3x.txt"<<endl; return -1;}

	Fmt1x fmt1;
	Fmt3x fmt3;
	fmt1.ReadMusicXML(string(argv[1]));
	fmt3.ConvertFromFmt1x(fmt1);
	fmt3.WriteFile(string(argv[2]));

	return 0;
}//end main
