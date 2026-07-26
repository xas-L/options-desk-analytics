#define _USE_MATH_DEFINES
#include <pybind11/pybind11.h>
#include <cmath>
#include <string>
#include <algorithm>
#include <cctype>

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
}
