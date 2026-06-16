#include "test.hpp"
#include "../inference/inference.h"
#include <cassert>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

int Test::argmax(std::vector<float>& v) 
{
    return std::distance(v.begin(),
        std::max_element(v.begin(), v.end()));
}

ClassId Test::classFromString(std::string& s) 
{
    if (s == "rabbit")  return ClassId::Rabbit;
    return ClassId::NotRabbit;
}

void Test::runStressTest(std::string& testDataPath)
{
    using clock = std::chrono::high_resolution_clock;

    int count = 0;
    double totalMs = 0.0;
    InferenceEngine engine;
    engine.loadModelInfo("model/model_info.json");
    if(engine.modelInfo.quantized)
    {
        torch::globalContext().setQEngine(at::QEngine::QNNPACK);

        torch::jit::setGraphExecutorOptimize(false);

        engine.loadModel("model/cgtat_q.pth");
    }
    else
    {
        engine.loadModel("model/cgtat.pth");
    }

    for (const auto& entry : fs::directory_iterator(testDataPath)) {
        if (!entry.is_regular_file())
            continue;

        std::string filename = entry.path().filename().string();
        std::string label = entry.path().stem().string(); // rabbit_01
        label = label.substr(0, label.find('_'));
        cv::Mat img = cv::imread(entry.path().string());
        if (img.empty()) {
            std::cerr << "Failed to load " << entry.path() << "\n";
            continue;
        }  
        
        auto t0 = clock::now();
        vector<float> confidenceScores = engine.run(img);
        auto t1 = clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        totalMs += ms;
        int predictedIdx = argmax(confidenceScores);
        ClassId predicted = static_cast<ClassId>(predictedIdx);

        ClassId groundTruth = classFromString(label);

        if(predicted == groundTruth)
        {
            count++;
        }

        std::cout << "[TEST] " << entry.path().filename()
                  << " -> " << ms << " ms"
                  << " predicted: " << ClassNames[predictedIdx] << "\n";
    }

    std::cout << "\n===== STRESS TEST RESULT =====\n";
    std::cout << "Hit rate: " << count << "\n";
    std::cout << "Avg inference: " << (totalMs / 10) << " ms\n";
    std::cout << "================================\n";

    assert(count > 0);
}
