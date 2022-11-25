#ifndef MIDI_HPP
#define MIDI_HPP

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
using namespace std;

inline char ch(int i){return i;}
inline int in(char c){return (int(c)+256)%256;}
inline string fnum(int i){
	stringstream ss;
	ss.str("");
	ss<<ch(i/16777216); i=i%16777216;//16777216=16^6
	ss<<ch(i/65536); i=i%65536;//65536=16^4
	ss<<ch(i/256); i=i%256;//256=16^2
	ss<<ch(i);
	return ss.str();
}//end fnum

inline string vnum(int i){
	stringstream ss;
	int b0,b1,b2,b3;
	b3=i/2097152;//2097152=2^21
	i=i%2097152;
	b2=i/16384;//16384=2^14
	i=i%16384;
	b1=i/128;//128=2^7
	i=i%128;
	b0=i;
	ss.str("");
	if(b3!=0){ss<<ch(128+b3);}
	if(b2!=0){ss<<ch(128+b2);}
	if(b1!=0){ss<<ch(128+b1);}
	ss<<ch(b0);
	return ss.str();
}//end vnum

inline int vnum2(int i){
	int d=0;
	int b0,b1,b2,b3;
	b3=i/2097152;//2097152=2^21
	i=i%2097152;
	b2=i/16384;//16384=2^14
	i=i%16384;
	b1=i/128;//128=2^7
	i=i%128;
	b0=i;
	if(b3!=0){d+=1;}
	if(b2!=0){d+=1;}
	if(b1!=0){d+=1;}
	d+=1;
	return d;
}//end vnum2

inline string tnum(int MM){
	stringstream ss;
	int i=60*1000000/MM;
	ss.str("");
	ss<<ch(i/65536); i=i%65536;//65536=16^4
	ss<<ch(i/256); i=i%256;//256=16^2
	ss<<ch(i);
	return ss.str();
}//end tnum


class MidiMessage{
public:
	MidiMessage(){}//end MidiMessage
	~MidiMessage(){}//end ~MidiMessage
	vector<int> mes;
	int tick;
	int track;
	double time;//time in sec calculated from the tempo indications

	bool isEffNoteOn() const {
		if(mes.size()!=3){return false;}
		if(mes[0]>=144 && mes[0]<160 && mes[2]>0){
			return true;
		}else{
			return false;
		}//endif
	}//end isEffNoteOn

	bool isEffNoteOff() const {
		if(mes.size()!=3){return false;}
		if(mes[0]>=128 && mes[0]<144){
			return true;
		}else if(mes[0]>=144 && mes[0]<160 && mes[2]==0){
			return true;
		}else{
			return false;
		}//endif
	}//end isEffNoteOn

};//end class MidiMessage

class LessTickMidiMessage{
public:
	bool operator()(const MidiMessage& a, const MidiMessage& b){
		return a.tick < b.tick;
	}//end operator()
};//end class LessTickMidiMessage

class LessMidiMessage{
public:
	bool operator()(const MidiMessage& a, const MidiMessage& b){
		if(a.time < b.time){
			return true;
		}else if(a.time==b.time){
			if(a.isEffNoteOff() && b.isEffNoteOn()){
				return true;
			}else{
				return false;
			}//endif
		}else{//if a.time > b.time
			return false;
		}//endif
	}//end operator()

};//end class LessMidiMessage
//sort(unifiedMes.begin(), unifiedMes.end(), LessMidiMessage());

class Midi{
public:
	vector<MidiMessage> content;
	int nTrack;
	int TPQN;
	int formatType;//=0 or 1
	vector<int> programChangeData;
	string strData;

	Midi(){programChangeData.assign(16,0);}//end Midi
	~Midi(){}//end ~Midi

	void ReadFile(string filename){

		content.clear();
		ifstream ifs;
		ifs.open(filename.c_str(),std::ios::binary);
		assert(ifs.is_open());
{
		char ch[100];
		ifs.read(ch,4);
		if(!(in(ch[0])==77&&in(ch[1])==84&&in(ch[2])==104&&in(ch[3])==100)){
			cout<<"Error: file does not start with MThd"<<endl;
			return;
		}//endif
		ifs.read(ch,4);// track length=6
		ifs.read(ch,2);//format number==0 or 1
		formatType=in(ch[1]);
//cout<<"Format type : "<<formatType<<endl;
		if(formatType!=0 && formatType!=1){cout<<"Error: format type is not 0 nor 1"<<endl;return;}
		ifs.read(ch,2);//number of track
		nTrack=in(ch[1]);
//cout<<"Number of tracks : "<<nTrack<<endl;
		if(nTrack<1){cout<<"Error: track number less than 1"<<endl;return;}
		ifs.read(ch,2);//tick per quater tone
		TPQN=in(ch[0])*16*16+in(ch[1]);
//cout<<"TPQN : "<<TPQN<<endl;
}//

		int runningStatus;
		for(int i=0;i<nTrack;i+=1){
			char pch[100];
			ifs.read(pch,4);
			if(!(in(pch[0])==77&&in(pch[1])==84&&in(pch[2])==114&&in(pch[3])==107)){
				cout<<"Error: track does not start with MTrk"<<endl;
				return;
			}//endif
			ifs.read(pch,4);// track length
			int trk_len=in(pch[3])+16*16*(in(pch[2])+16*16*(in(pch[1])+16*16*in(pch[0])));
			// char ch[trk_len];
			vector<char> ch(trk_len);
			// ifs.read(ch,trk_len);// track data
			ifs.read(ch.data(),trk_len);// track data

/*
cout<<"# track "<<i<<endl;
for(int n=0;n<trk_len;n+=1){
cout<<n<<" "<<in(ch[n])<<endl;
}//endfor n
*/

			int curByte=0;
			int tick=0;//cumulative tick
			int deltaTick=0;
			bool readTick=true;
			MidiMessage midiMes;
			midiMes.track=i;
			vector<int> vi;
			while(curByte<trk_len){

				if(readTick){
					deltaTick=0;
					while(true){
						if(in(ch[curByte])<128){break;}
						deltaTick=128*deltaTick+(in(ch[curByte])-128);
						curByte+=1;
					}//endwhile
					deltaTick=128*deltaTick+in(ch[curByte]);
					tick+=deltaTick;
					readTick=false; curByte+=1; continue;
				}//endif

				vi.clear();
				if((in(ch[curByte])>=128 && in(ch[curByte])<192) || (in(ch[curByte])>=224 && in(ch[curByte])<240)){
					runningStatus=in(ch[curByte]);
					vi.push_back( in(ch[curByte]) ); vi.push_back( in(ch[curByte+1]) ); vi.push_back( in(ch[curByte+2]) );
					curByte+=3;
				}else if(in(ch[curByte])>=192&&in(ch[curByte])<224){
					runningStatus=in(ch[curByte]);
					vi.push_back( in(ch[curByte]) ); vi.push_back( in(ch[curByte+1]) );
					curByte+=2;
				}else if(in(ch[curByte])==240 || in(ch[curByte])==255){
					runningStatus=in(ch[curByte]);
					vi.push_back( in(ch[curByte]) );
					curByte+=1;
					if(runningStatus==255){
						vi.push_back( in(ch[curByte]) );//type of metaevent
						curByte+=1;
					}//endif
					int numBytes=0;
					while(true){
						if(in(ch[curByte])<128) break;
						numBytes=128*numBytes+(in(ch[curByte])-128);
						vi.push_back( in(ch[curByte]) );
						curByte+=1;
					}//endwhile
					numBytes=128*numBytes+in(ch[curByte]);
					vi.push_back( in(ch[curByte]) );
					for(int j=0;j<numBytes;j+=1){vi.push_back( in(ch[curByte+1+j]) );}
					curByte+=1+numBytes;
				}else if(in(ch[curByte])<128){
					if((runningStatus>=128 && runningStatus<192) || (runningStatus>=224 && runningStatus<240)){
						vi.push_back( runningStatus ); vi.push_back( in(ch[curByte]) ); vi.push_back( in(ch[curByte+1]) );
						curByte+=2;
					}else if(runningStatus>=192 && runningStatus<224){
						vi.push_back( runningStatus ); vi.push_back( in(ch[curByte]) );
						curByte+=1;
					}else if(runningStatus==240 || runningStatus==255){
						vi.push_back( runningStatus );
						if(runningStatus==255){curByte+=1; vi.push_back( in(ch[curByte]) );}
						curByte+=1;
						int numBytes=0;
						while(true){
							if(in(ch[curByte])<128) break;
							numBytes=128*numBytes+(in(ch[curByte])-128);
							vi.push_back( in(ch[curByte]) );
							curByte+=1;
						}//endwhile
						numBytes=128*numBytes+in(ch[curByte]);
						vi.push_back( in(ch[curByte]) );
						for(int j=0;j<numBytes;j+=1){vi.push_back( in(ch[curByte+1+j]) );}
						curByte+=1+numBytes;
					}else{
						cout<<"Error: runningStatus has an invalid value : "<<runningStatus<<endl; return; break;
					}//endif
				}else{
					cout<<"Error: unknown events in the trk : "<<i<<" "<<curByte<<" "<<in(ch[curByte])<<endl; return; break;
				}//endif

				midiMes.tick=tick;
				midiMes.mes=vi;
				content.push_back(midiMes);
				readTick=true;

			}//endwhile

		}//endfor i
		ifs.close();

		stable_sort(content.begin(), content.end(), LessTickMidiMessage());

////// set times
		int tickPoint=0;
		double timePoint=0;
		int currDurQN=500000;
		for(int i=0;i<content.size();i+=1){
			content[i].time=timePoint+((double)(content[i].tick-tickPoint)/(double)TPQN)*((double)currDurQN/1000000.);
			if(content[i].mes[0]==255 && content[i].mes[1]==81 && content[i].mes[2]==3){
				currDurQN=content[i].mes[3]*256*256+content[i].mes[4]*256+content[i].mes[5];
				timePoint=content[i].time;
				tickPoint=content[i].tick;
			}//endif
		}//endfor i

	}//end ReadFile

	void OrderContent(){
		stable_sort(content.begin(), content.end(), LessTickMidiMessage());
	}//end OrderContent

	void WriteNoteOn(string outputFile){
		ofstream ofs(outputFile.c_str());
		for(int i=0;i<content.size();i+=1){
			if(content[i].mes[0]>=144 && content[i].mes[0]<160){//note-on event
				if(content[i].mes[2]>0){//truely note-on event
ofs<<content[i].time<<"\t"<<content[i].mes.size()<<"\t";
					for(int j=0;j<content[i].mes.size();j+=1){
ofs<<content[i].mes[j]<<"\t";
					}//endfor j
ofs<<"\n";
				}//endif
			}//endif
		}//endfor i
		ofs.close();
	}//end WriteNoteOn

	void SetProgramChangeData(vector<int> data){
		for(int i=0;i<16;i+=1){
			programChangeData[i]=data[i];
		}//endfor i
	}//end SetProgramChangeData

	void SetStrData(){
		stringstream ss,ss1;
		int nBytes;
		double timeInterval;
		ss1.str(""); nBytes=0;

		for(int i=0;i<16;i+=1){
			ss1<<ch(0)<<ch(192+i)<<ch(programChangeData[i]);
			nBytes+=3;
		}//endfor i

		for(int i=0;i<content.size();i+=1){
			if(i==0){
				timeInterval=content[i].time;
			}else{
				timeInterval=content[i].time-content[i-1].time;
			}//endif
			ss1<<vnum( (int)(5040*timeInterval) );
			nBytes+=vnum2( (int)(5040*timeInterval) );
			for(int j=0;j<content[i].mes.size();j+=1){
				ss1<<ch(content[i].mes[j]);
				nBytes+=1;
			}//endfor j
		}//endfor i

		ss.str("");
		ss<<"MThd";
		ss<<fnum(6);//number of bytes
		ss<<ch(0)<<ch(0);//format type
		ss<<ch(0)<<ch(1);//number of tracks
//		ss<<ch(3)<<ch(192);// 960 resolution <=4095
		ss<<ch(9)<<ch(216);// 2520 resolution <=4095

		ss<<"MTrk";//Track 1
		ss<<fnum(7+nBytes+4);//number of bytes
		ss<<ch(0)<<ch(255)<<ch(81)<<ch(3);//set tempo
		int tempo=120;
		ss<<tnum(tempo);//tempo=120
		ss<<ss1.str();
		ss<<ch(0)<<ch(255)<<ch(47)<<ch(0);//Track end

		strData=ss.str();

	}//end SetStrData

};//end class Midi

#endif // MIDI_HPP

