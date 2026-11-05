#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace disparity
{
    constexpr std::size_t input_count = 11;
    constexpr std::size_t hidden_count = 10;
    constexpr std::size_t weight_count = input_count * hidden_count + hidden_count;
    using Patch = std::array<double, input_count>;
    using Weights = std::array<double, weight_count>;

    struct Evaluation
    {
        double merit = 0.0;
        double short_variance = 0.0;
        double long_variance = 0.0;
        Weights gradient{};
        std::vector<double> predictions;
    };

    struct Epoch
    {
        unsigned iteration;
        double merit;
        double short_variance;
        double long_variance;
        double gradient_norm;
        double step;
    };

    std::vector<double> read_numbers(const std::string& path);
    std::vector<Patch> make_patches(const std::vector<double>& left,
                                    const std::vector<double>& right,
                                    std::size_t stride = 7);
    void write_numbers(const std::string& path, const std::string& header,
                       const std::vector<double>& values);
    double correlation(const std::vector<double>& a, const std::vector<double>& b);

    class Network
    {
    public:
        Network(std::vector<Patch> samples, double short_half_life = 32.0,
                double long_half_life = 3200.0);
        static Weights initial_weights(std::uint32_t seed);
        Evaluation evaluate(const Weights& weights, bool with_gradient = true) const;
        std::vector<Epoch> train(Weights& weights, unsigned steps,
                                 const std::function<void(unsigned, const Weights&)>& checkpoint = {}) const;

    private:
        std::vector<Patch> samples_;
        std::vector<double> short_kernel_;
        std::vector<double> long_kernel_;
        std::vector<double> convolve(const std::vector<double>& input,
                                     const std::vector<double>& kernel) const;
    };
}
