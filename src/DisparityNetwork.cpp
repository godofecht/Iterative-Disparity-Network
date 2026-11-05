#include "DisparityNetwork.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace disparity
{
    namespace
    {
        constexpr double epsilon = 1e-12;

        std::vector<double> kernel(std::size_t n, double half_life)
        {
            if (!(half_life > 0.0) || !std::isfinite(half_life))
                throw std::invalid_argument("A half-life must be finite and positive");
            std::vector<double> k(n);
            double total = 0.0;
            for (std::size_t i = 0; i < n; ++i)
            {
                const auto distance = std::min(i, n - i);
                k[i] = std::exp(-std::log(2.0) * static_cast<double>(distance) / half_life);
                total += k[i];
            }
            for (double& value : k)
                value /= total;
            return k;
        }

        double dot(const Weights& a, const Weights& b)
        {
            double result = 0.0;
            for (std::size_t i = 0; i < weight_count; ++i)
                result += a[i] * b[i];
            return result;
        }
    }

    std::vector<double> read_numbers(const std::string& path)
    {
        std::ifstream stream(path);
        if (!stream)
            throw std::runtime_error("Cannot open: " + path);
        std::vector<double> result;
        std::string token;
        while (stream >> token)
        {
            if (token == "Values" || token == "value" || token == "prediction" || token == "weight")
                continue;
            // A CSV row is a whitespace-separated sequence or a single numeric column.
            if (!token.empty() && token.back() == ',')
                token.pop_back();
            std::size_t used = 0;
            double value;
            try
            {
                value = std::stod(token, &used);
            }
            catch (const std::exception&)
            {
                throw std::runtime_error("Invalid number in " + path + ": " + token);
            }
            if (used != token.size() || !std::isfinite(value))
                throw std::runtime_error("Non-finite or invalid number in " + path + ": " + token);
            result.push_back(value);
        }
        if (result.empty() || !stream.eof())
            throw std::runtime_error("Empty or unreadable input: " + path);
        return result;
    }

    std::vector<Patch> make_patches(const std::vector<double>& left,
                                    const std::vector<double>& right,
                                    std::size_t stride)
    {
        if (left.size() != right.size() || left.size() < 5 || stride < 5)
            throw std::invalid_argument("Equal stereo sequences of at least five pixels and stride >= 5 are required");
        if (left.size() % stride != 0)
            throw std::invalid_argument("Input length must be divisible by the sampling stride");
        const auto n = left.size() / stride;
        std::vector<Patch> result;
        result.reserve(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            const auto offset = i * stride;
            Patch patch{};
            for (std::size_t j = 0; j < 5; ++j)
            {
                patch[j] = left[offset + j];
                patch[5 + j] = right[offset + j];
            }
            patch[10] = 1.0;
            result.push_back(patch);
        }
        return result;
    }

    void write_numbers(const std::string& path, const std::string& header,
                       const std::vector<double>& values)
    {
        std::ofstream stream(path);
        if (!stream)
            throw std::runtime_error("Cannot write: " + path);
        stream << header << '\n' << std::setprecision(17);
        for (const double value : values)
            stream << value << '\n';
        if (!stream)
            throw std::runtime_error("Failed to write: " + path);
    }

    double correlation(const std::vector<double>& a, const std::vector<double>& b)
    {
        if (a.size() != b.size() || a.size() < 2)
            throw std::invalid_argument("Correlation needs equal-length nonempty sequences");
        const double mean_a = std::accumulate(a.begin(), a.end(), 0.0) / a.size();
        const double mean_b = std::accumulate(b.begin(), b.end(), 0.0) / b.size();
        double covariance = 0.0, variance_a = 0.0, variance_b = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            const double da = a[i] - mean_a, db = b[i] - mean_b;
            covariance += da * db;
            variance_a += da * da;
            variance_b += db * db;
        }
        if (variance_a == 0.0 || variance_b == 0.0)
            throw std::domain_error("Correlation undefined for constant data");
        return covariance / std::sqrt(variance_a * variance_b);
    }

    Network::Network(std::vector<Patch> samples, double short_half_life,
                     double long_half_life)
        : samples_(std::move(samples))
    {
        if (samples_.size() < 3)
            throw std::invalid_argument("At least three stereo patches are required");
        if (!(short_half_life < long_half_life))
            throw std::invalid_argument("Short half-life must be smaller than long half-life");
        short_kernel_ = kernel(samples_.size(), short_half_life);
        long_kernel_ = kernel(samples_.size(), long_half_life);
    }

    Weights Network::initial_weights(std::uint32_t seed)
    {
        std::mt19937 engine(seed);
        std::normal_distribution<double> normal(0.0, 0.12);
        Weights weights{};
        for (auto& weight : weights)
            weight = normal(engine);
        return weights;
    }

    std::vector<double> Network::convolve(const std::vector<double>& values,
                                          const std::vector<double>& k) const
    {
        const std::size_t n = values.size();
        std::vector<double> output(n, 0.0);
        for (std::size_t i = 0; i < n; ++i)
        {
            double sum = 0.0;
            // Circular convolution: the corresponding kernel is symmetric.
            for (std::size_t j = 0; j < n; ++j)
                sum += k[j >= i ? j - i : n + j - i] * values[j];
            output[i] = sum;
        }
        return output;
    }

    Evaluation Network::evaluate(const Weights& w, bool with_gradient) const
    {
        Evaluation result;
        const std::size_t n = samples_.size();
        result.predictions.reserve(n);
        std::vector<std::array<double, hidden_count>> hidden;
        if (with_gradient)
            hidden.reserve(n);
        for (const auto& patch : samples_)
        {
            std::array<double, hidden_count> h{};
            double y = 0.0;
            for (std::size_t j = 0; j < hidden_count; ++j)
            {
                double activation = 0.0;
                for (std::size_t i = 0; i < input_count; ++i)
                    activation += patch[i] * w[j * input_count + i];
                h[j] = std::tanh(activation);
                y += h[j] * w[input_count * hidden_count + j];
            }
            result.predictions.push_back(y);
            if (with_gradient)
                hidden.push_back(h);
        }

        const auto short_average = convolve(result.predictions, short_kernel_);
        const auto long_average = convolve(result.predictions, long_kernel_);
        std::vector<double> short_residual(n), long_residual(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            short_residual[i] = short_average[i] - result.predictions[i];
            long_residual[i] = long_average[i] - result.predictions[i];
            result.short_variance += 0.5 * short_residual[i] * short_residual[i];
            result.long_variance += 0.5 * long_residual[i] * long_residual[i];
        }
        result.merit = std::log((result.long_variance + epsilon) /
                                (result.short_variance + epsilon));
        if (!std::isfinite(result.merit))
            throw std::runtime_error("Non-finite objective");
        if (!with_gradient)
            return result;

        // For E = 1/2 ||(K-I)y||^2, dE/dy = (K-I)^T (K-I)y.
        // Symmetric circular kernels imply K^T = K.
        const auto short_adj = convolve(short_residual, short_kernel_);
        const auto long_adj = convolve(long_residual, long_kernel_);
        for (std::size_t k = 0; k < n; ++k)
        {
            const double dy = (long_adj[k] - long_residual[k]) /
                                   (result.long_variance + epsilon) -
                              (short_adj[k] - short_residual[k]) /
                                   (result.short_variance + epsilon);
            for (std::size_t h = 0; h < hidden_count; ++h)
            {
                const double activation = hidden[k][h];
                result.gradient[input_count * hidden_count + h] += dy * activation;
                const double upstream = dy * w[input_count * hidden_count + h] *
                                        (1.0 - activation * activation);
                for (std::size_t j = 0; j < input_count; ++j)
                    result.gradient[h * input_count + j] += upstream * samples_[k][j];
            }
        }
        for (double g : result.gradient)
            if (!std::isfinite(g))
                throw std::runtime_error("Non-finite gradient");
        return result;
    }

    std::vector<Epoch> Network::train(Weights& weights, unsigned steps,
                                       const std::function<void(unsigned, const Weights&)>& checkpoint) const
    {
        auto current = evaluate(weights);
        Weights direction = current.gradient;
        std::vector<Epoch> history;
        history.reserve(steps + 1);
        history.push_back({0, current.merit, current.short_variance,
                           current.long_variance, std::sqrt(dot(current.gradient, current.gradient)), 0.0});
        if (checkpoint)
            checkpoint(0, weights);
        for (unsigned iteration = 1; iteration <= steps; ++iteration)
        {
            if (dot(current.gradient, current.gradient) < 1e-16)
                break;
            double slope = dot(current.gradient, direction);
            if (slope <= 0.0 || !std::isfinite(slope))
            {
                direction = current.gradient;
                slope = dot(current.gradient, direction);
            }
            double step = 1.0;
            Weights candidate_weights{};
            Evaluation candidate;
            bool accepted = false;
            while (step > 1e-13)
            {
                for (std::size_t k = 0; k < weight_count; ++k)
                    candidate_weights[k] = weights[k] + step * direction[k];
                candidate = evaluate(candidate_weights, false);
                if (candidate.merit >= current.merit + 1e-4 * step * slope)
                {
                    accepted = true;
                    break;
                }
                step *= 0.5;
            }
            if (!accepted)
                break;
            weights = candidate_weights;
            auto next = evaluate(weights);
            const double old_norm_squared = dot(current.gradient, current.gradient);
            double numerator = 0.0;
            for (std::size_t i = 0; i < weight_count; ++i)
                numerator += next.gradient[i] * (next.gradient[i] - current.gradient[i]);
            const double beta = std::max(0.0, numerator / old_norm_squared);
            for (std::size_t i = 0; i < weight_count; ++i)
                direction[i] = next.gradient[i] + beta * direction[i];
            current = std::move(next);
            history.push_back({iteration, current.merit, current.short_variance,
                               current.long_variance, std::sqrt(dot(current.gradient, current.gradient)), step});
            if (checkpoint && (iteration % 50 == 0 || iteration == steps))
                checkpoint(iteration, weights);
        }
        return history;
    }
}
