#pragma once
#include <iostream>

using namespace std;

class PointInfo {

    private:
        int x;
        int y;
        double result;

    public:
        PointInfo(int x, int y, double result) : x(x), y(y), result(result) {};
        //PointInfo(const PointInfo&) = default;
        //PointInfo& operator=(const PointInfo&) = default;

        ~PointInfo() {
            //cout << "PointInfo destructor called" << endl;
        };
        
        int getX() const { return this->x; };
        int getY() const { return this->y; };
        double getResult() const { return this->result; };
};