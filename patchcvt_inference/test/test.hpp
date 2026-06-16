#ifndef TEST_HPP_
#define TEST_HPP_

#include <string>
#include <vector>

using namespace std;

enum class ClassId : int {
    Rabbit  = 0,
    NotRabbit = 1
};

class Test
{
    public:
        void runStressTest(std::string& testDataPath);
    private:
        int argmax(std::vector<float>& v);
        ClassId classFromString(std:: string& s);
    
        std::vector<std::string> ClassNames = {
            "rabbit",
            "not_rabbit"
        };
};

#endif