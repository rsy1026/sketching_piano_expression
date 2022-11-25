#include<fstream>
#include<iostream>
#include<cmath>
#include<string>
#include<sstream>
#include<vector>
#include<algorithm>
#include"stdio.h"
#include"stdlib.h"
#include"ScoreFollower_v170101_2.hpp"
using namespace std;

int main(int argc, char** argv){
	vector<int> v(100);
	vector<double> d(100);
	vector<string> s(100);
	stringstream ss;
	clock_t start, end;
	start = clock();

	if(argc!=5){
		cout<<"Error in usage: $./ScorePerfmMatcher hmm.txt perfm_spr.txt result_match.txt secPerQuarterNote(=0.01)"<<endl;
		return -1;
	}//endif
	string hmmName=string(argv[1]);
	string perfmName=string(argv[2]);
	string matchfileName=string(argv[3]);
	double secPerQN=atof(argv[4]);

	ScoreFollower scofo(hmmName,secPerQN);

	PianoRoll pr;
	pr.ReadFileSpr(perfmName);

	ScorePerfmMatch match;
	match=scofo.GetMatchResult(pr);

	match.comments.push_back(" Score: "+hmmName);
	match.comments.push_back(" Perfm: "+perfmName);

	match.WriteFile(matchfileName);

//	end = clock(); cout<<"Elapsed time : "<<((double)(end - start) / CLOCKS_PER_SEC)<<" sec"<<endl; start=end;
	return 0;
}//end main

