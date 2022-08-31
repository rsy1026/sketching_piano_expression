#include<iostream>
#include<iomanip>
#include<fstream>
#include<string>
#include<sstream>
#include<vector>
#include<stdio.h>
#include<stdlib.h>
#include<cmath>
#include<cassert>
#include<algorithm>
#include "PianoRoll_v170117.hpp"
using namespace std;

int main(int argc, char** argv) {
	vector<int> v(100);
	vector<double> d(100);
	vector<string> s(100);
	stringstream ss;

	if(argc!=3){cout<<"Error in usage: $./this (0:spr/1:ipr/2:spr[pedal on]/3:ipr[pedal on]) in(.mid)"<<endl; return -1;}
		string infileStem=string(argv[2]);
		int pianoRollType=atoi(argv[1]);//0: spelled pitch, 1: integral pitch
		if(pianoRollType<0 || pianoRollType>3){
		cout<<"Error in usage: $./this (0:spr/1:ipr/2:spr[pedal on]/3:ipr[pedal on]) in(.mid)"<<endl; return -1;
	}//endif

	PianoRoll pr;
	ss.str(""); ss<<infileStem<<".mid";
	pr.ReadMIDIFile(ss.str());

	if(pianoRollType==0||pianoRollType==2){
		ss.str(""); ss<<infileStem<<"_spr.txt";
		pr.WriteFileSpr(ss.str(),(pianoRollType==2));
	}else if(pianoRollType==1||pianoRollType==3){
		ss.str(""); ss<<infileStem<<"_ipr.txt";
		pr.WriteFileIpr(ss.str(),(pianoRollType==3));
	}//endif

  return 0;
}//end main


