#include "DisparityNetwork.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace
{
    void check(bool value, const char* message)
    {
        if (!value)
            throw std::runtime_error(message);
    }
}

int main()
{
    using namespace disparity;
    std::vector<double> left(7 * 24), right(7 * 24);
    for (std::size_t i = 0; i < left.size(); ++i)
    {
        left[i] = std::sin(0.32 * static_cast<double>(i)) + 0.2 * std::cos(0.12 * i);
        right[i] = std::sin(0.32 * static_cast<double>(i) + 0.08) + 0.2 * std::cos(0.12 * i);
    }
    const auto samples = make_patches(left, right);
    check(samples.size() == 24, "Patch count");
    check(samples[0][0] == left[0] && samples[1][0] == left[7], "Patch stride");
    check(samples[0][5] == right[0] && samples[0][10] == 1.0, "Stereo input/bias");
    const Network model(samples, 2.0, 100.0);
    auto w = Network::initial_weights(12);
    check(w == Network::initial_weights(12), "Deterministic seeding");
    auto eval = model.evaluate(w);
    check(std::isfinite(eval.merit) && eval.predictions.size() == 24, "Finite objective");
    double max_abs_error = 0.0;
    for (std::size_t k = 0; k < weight_count; ++k)
    {
        const double original = w[k];
        const double h = 1e-5;
        w[k] = original + h;
        const double fp = model.evaluate(w, false).merit;
        w[k] = original - h;
        const double fm = model.evaluate(w, false).merit;
        w[k] = original;
        const double numeric = (fp - fm) / (2.0 * h);
        const double tolerance = 1e-4 * (1.0 + std::abs(numeric));
        max_abs_error = std::max(max_abs_error, std::abs(numeric - eval.gradient[k]));
        check(std::abs(numeric - eval.gradient[k]) < tolerance, "Analytic gradient differs from finite differences");
    }
    const auto old = eval.merit;
    const auto history = model.train(w, 12);
    check(history.size() >= 2, "No accepted optimization steps");
    check(history.back().merit > old, "Conjugate gradient did not increase the objective");
    for (std::size_t i = 1; i < history.size(); ++i)
        check(history[i].merit >= history[i - 1].merit, "Objective must be monotonic");
    check(std::abs(correlation({1, 2, 3}, {3, 2, 1}) + 1.0) < 1e-12, "Correlation negative sign");
    bool rejected = false;
    try { make_patches({1, 2, 3}, {1, 2, 3}); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "Must reject invalid samples");
    std::cout << "passed; 120 gradient coordinates; maximum error=" << max_abs_error
              << "; initial merit=" << old << "; final merit=" << history.back().merit << '\n';
}
