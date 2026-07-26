#define _USE_MATH_DEFINES
#include <pybind11/pybind11.h>
#include <cmath>
#include <string>
#include <algorithm>
#include <cctype>
#include <complex>
#include <vector>
#include <pybind11/stl.h>

namespace py = pybind11;

// Standard normal CDF
inline double norm_cdf(double x) {
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}

// Standard normal PDF
inline double norm_pdf(double x) {
    return (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * x * x);
}

// Convert string to lower case for case-insensitive option type checks
inline std::string to_lower(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return result;
}

// Black-Scholes price
double bs_price(double S, double K, double T, double r, double sigma, const std::string& option_type, double q) {
    std::string type = to_lower(option_type);
    bool is_call = (type == "call" || type == "c");
    
    double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    
    if (is_call) {
        return S * std::exp(-q * T) * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
    } else {
        return K * std::exp(-r * T) * norm_cdf(-d2) - S * std::exp(-q * T) * norm_cdf(-d1);
    }
}

// Black-Scholes delta
double bs_delta(double S, double K, double T, double r, double sigma, const std::string& option_type, double q) {
    std::string type = to_lower(option_type);
    bool is_call = (type == "call" || type == "c");
    
    double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    
    if (is_call) {
        return std::exp(-q * T) * norm_cdf(d1);
    } else {
        return std::exp(-q * T) * (norm_cdf(d1) - 1.0);
    }
}

// Black-Scholes gamma
double bs_gamma(double S, double K, double T, double r, double sigma, double q) {
    double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    return std::exp(-q * T) * norm_pdf(d1) / (S * sigma * std::sqrt(T));
}

// Black-Scholes vega
double bs_vega(double S, double K, double T, double r, double sigma, double q) {
    double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    return S * std::exp(-q * T) * norm_pdf(d1) * std::sqrt(T);
}

// Black-Scholes theta
double bs_theta(double S, double K, double T, double r, double sigma, const std::string& option_type, double q) {
    std::string type = to_lower(option_type);
    bool is_call = (type == "call" || type == "c");
    
    double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    
    double term1 = -(S * std::exp(-q * T) * norm_pdf(d1) * sigma) / (2.0 * std::sqrt(T));
    
    if (is_call) {
        return term1 - r * K * std::exp(-r * T) * norm_cdf(d2) + q * S * std::exp(-q * T) * norm_cdf(d1);
    } else {
        return term1 + r * K * std::exp(-r * T) * norm_cdf(-d2) - q * S * std::exp(-q * T) * norm_cdf(-d1);
    }
}

// Black-Scholes rho
double bs_rho(double S, double K, double T, double r, double sigma, const std::string& option_type, double q) {
    std::string type = to_lower(option_type);
    bool is_call = (type == "call" || type == "c");
    
    double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
    double d2 = d1 - sigma * std::sqrt(T);
    
    if (is_call) {
        return K * T * std::exp(-r * T) * norm_cdf(d2);
    } else {
        return -K * T * std::exp(-r * T) * norm_cdf(-d2);
    }
}

const std::complex<double> I_COMPLEX(0.0, 1.0);

// Basic Radix-2 Cooley-Tukey FFT
void fft(std::vector<std::complex<double>>& x) {
    size_t n = x.size();
    if (n <= 1) return;

    std::vector<std::complex<double>> even(n / 2), odd(n / 2);
    for (size_t i = 0; i < n / 2; ++i) {
        even[i] = x[i * 2];
        odd[i] = x[i * 2 + 1];
    }

    fft(even);
    fft(odd);

    for (size_t i = 0; i < n / 2; ++i) {
        std::complex<double> t = std::polar(1.0, -2.0 * M_PI * i / n) * odd[i];
        x[i] = even[i] + t;
        x[i + n / 2] = even[i] - t;
    }
}

// Heston characteristic function
std::complex<double> heston_cf(std::complex<double> u, double S0, double r, double q, double T,
                               double kappa, double theta, double sigma, double rho, double V0) {
    std::complex<double> a = kappa - I_COMPLEX * rho * sigma * u;
    std::complex<double> d = std::sqrt(a * a + sigma * sigma * (u * u + I_COMPLEX * u));
    std::complex<double> g = (a - d) / (a + d);

    std::complex<double> exp_d = std::exp(-d * T);
    std::complex<double> C = (kappa * theta / (sigma * sigma)) * 
                             ((a - d) * T - 2.0 * std::log((1.0 - g * exp_d) / (1.0 - g)));
    std::complex<double> D = (a - d) / (sigma * sigma) * ((1.0 - exp_d) / (1.0 - g * exp_d));

    return std::exp(I_COMPLEX * u * (std::log(S0) + (r - q) * T) + C + D * V0);
}

// Single Heston FFT Price
double heston_fft_price(double S0, double K, double T, double r, double q,
                        double kappa, double theta, double sigma, double rho, double V0,
                        const std::string& option_type = "call",
                        int N = 4096, double eta = 0.25, double alpha = 1.5) {
    
    std::string type = to_lower(option_type);
    bool is_call = (type == "call" || type == "c");

    double lambda_ = 2.0 * M_PI / (N * eta);
    double b = (N * lambda_) / 2.0;

    std::vector<std::complex<double>> x(N);
    for (int j = 0; j < N; ++j) {
        double u_val = j * eta;
        std::complex<double> u(u_val, 0.0);
        std::complex<double> u_mod = u - (alpha + 1.0) * I_COMPLEX;
        
        std::complex<double> cf = heston_cf(u_mod, S0, r, q, T, kappa, theta, sigma, rho, V0);
        std::complex<double> psi = std::exp(-r * T) * cf / (alpha * alpha + alpha - u_val * u_val + I_COMPLEX * (2.0 * alpha + 1.0) * u_val);

        double w = (j == 0 || j == N - 1) ? 1.0 / 3.0 : (j % 2 == 0 ? 2.0 / 3.0 : 4.0 / 3.0);
        x[j] = std::exp(I_COMPLEX * b * u_val) * psi * eta * w;
    }

    fft(x);

    // Find closest strike index
    double k_val = std::log(K);
    double exact_idx = (k_val + b) / lambda_;
    int idx1 = std::max(0, std::min(N - 2, static_cast<int>(std::floor(exact_idx))));
    int idx2 = idx1 + 1;
    
    double k1 = -b + idx1 * lambda_;
    double k2 = -b + idx2 * lambda_;

    double price1 = std::exp(-alpha * k1) / M_PI * x[idx1].real();
    double price2 = std::exp(-alpha * k2) / M_PI * x[idx2].real();

    double call_price = price1 + (price2 - price1) * (k_val - k1) / (k2 - k1);

    if (is_call) {
        return call_price;
    } else {
        return call_price - S0 * std::exp(-q * T) + K * std::exp(-r * T);
    }
}

// Batch Heston FFT Pricing
std::vector<double> heston_fft_price_batch(double S0, const std::vector<double>& K_vec, double T, double r, double q,
                                           double kappa, double theta, double sigma, double rho, double V0,
                                           const std::string& option_type = "call",
                                           int N = 4096, double eta = 0.25, double alpha = 1.5) {
    std::string type = to_lower(option_type);
    bool is_call = (type == "call" || type == "c");

    double lambda_ = 2.0 * M_PI / (N * eta);
    double b = (N * lambda_) / 2.0;

    std::vector<std::complex<double>> x(N);
    for (int j = 0; j < N; ++j) {
        double u_val = j * eta;
        std::complex<double> u(u_val, 0.0);
        std::complex<double> u_mod = u - (alpha + 1.0) * I_COMPLEX;
        
        std::complex<double> cf = heston_cf(u_mod, S0, r, q, T, kappa, theta, sigma, rho, V0);
        std::complex<double> psi = std::exp(-r * T) * cf / (alpha * alpha + alpha - u_val * u_val + I_COMPLEX * (2.0 * alpha + 1.0) * u_val);

        double w = (j == 0 || j == N - 1) ? 1.0 / 3.0 : (j % 2 == 0 ? 2.0 / 3.0 : 4.0 / 3.0);
        x[j] = std::exp(I_COMPLEX * b * u_val) * psi * eta * w;
    }

    fft(x);

    std::vector<double> prices;
    prices.reserve(K_vec.size());
    for (double K : K_vec) {
        double k_val = std::log(K);
        double exact_idx = (k_val + b) / lambda_;
        int idx1 = std::max(0, std::min(N - 2, static_cast<int>(std::floor(exact_idx))));
        int idx2 = idx1 + 1;
        
        double k1 = -b + idx1 * lambda_;
        double k2 = -b + idx2 * lambda_;

        double price1 = std::exp(-alpha * k1) / M_PI * x[idx1].real();
        double price2 = std::exp(-alpha * k2) / M_PI * x[idx2].real();

        double call_price = price1 + (price2 - price1) * (k_val - k1) / (k2 - k1);

        if (is_call) {
            prices.push_back(call_price);
        } else {
            prices.push_back(call_price - S0 * std::exp(-q * T) + K * std::exp(-r * T));
        }
    }
    return prices;
}

PYBIND11_MODULE(bs_pricer_cpp, m) {
    m.doc() = "Native C++ Black-Scholes pricer and Greeks.";
    
    m.def("bs_price", &bs_price, "Calculate Black-Scholes price.",
          py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), py::arg("sigma"),
          py::arg("option_type") = "call", py::arg("q") = 0.0);
          
    m.def("bs_delta", &bs_delta, "Calculate Black-Scholes delta.",
          py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), py::arg("sigma"),
          py::arg("option_type") = "call", py::arg("q") = 0.0);
          
    m.def("bs_gamma", &bs_gamma, "Calculate Black-Scholes gamma.",
          py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), py::arg("sigma"), py::arg("q") = 0.0);
          
    m.def("bs_vega", &bs_vega, "Calculate Black-Scholes vega.",
          py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), py::arg("sigma"), py::arg("q") = 0.0);
          
    m.def("bs_theta", &bs_theta, "Calculate Black-Scholes theta.",
          py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), py::arg("sigma"),
          py::arg("option_type") = "call", py::arg("q") = 0.0);
          
    m.def("bs_rho", &bs_rho, "Calculate Black-Scholes rho.",
          py::arg("S"), py::arg("K"), py::arg("T"), py::arg("r"), py::arg("sigma"),
          py::arg("option_type") = "call", py::arg("q") = 0.0);
          
    m.def("heston_fft_price", &heston_fft_price, "Calculate Heston price via FFT.",
          py::arg("S0"), py::arg("K"), py::arg("T"), py::arg("r"), py::arg("q"),
          py::arg("kappa"), py::arg("theta"), py::arg("sigma"), py::arg("rho"), py::arg("V0"),
          py::arg("option_type") = "call", py::arg("N") = 4096, py::arg("eta") = 0.25, py::arg("alpha") = 1.5);
          
    m.def("heston_fft_price_batch", &heston_fft_price_batch, "Calculate Heston price via FFT for multiple strikes.",
          py::arg("S0"), py::arg("K_vec"), py::arg("T"), py::arg("r"), py::arg("q"),
          py::arg("kappa"), py::arg("theta"), py::arg("sigma"), py::arg("rho"), py::arg("V0"),
          py::arg("option_type") = "call", py::arg("N") = 4096, py::arg("eta") = 0.25, py::arg("alpha") = 1.5);
}
