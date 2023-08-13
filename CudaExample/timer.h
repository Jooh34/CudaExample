#ifndef __Timer__
#define __Timer__

#include <vector>
#include <string>
#include <chrono>

using namespace std;
using namespace chrono;

class Timer
{
private:
	static Timer* instance;

public:
	static Timer& getInstance() {
		if (!instance) {
			instance = new Timer();
		}
		
		return *instance;
	}

	void addRecord(string name, double time) {
		names.push_back(name);
		counters.push_back(time);
	}

	void printRecord() {
		int size = names.size();
		printf("---- Timer ----\n");
		for (int i = 0; i < size; i++) {
			printf("%s - %lf ms\n", names[i].c_str(), counters[i]);
		}
	}
	vector<string> names;
	vector<double> counters;
};
Timer* Timer::instance = nullptr;

#define SCOPED_TIMER(Name) ScopedTimer t1 = ScopedTimer(Name);

class ScopedTimer
{
private:
	string name;
	steady_clock::time_point start;

public:
	ScopedTimer(string inName)
	{
		start = steady_clock::now();
		name = inName;
	}

	~ScopedTimer()
	{
		duration<double> sec = (steady_clock::now() - start);
		Timer::getInstance().addRecord(name, sec.count() * 1000);
	}
};

#endif // __Timer__