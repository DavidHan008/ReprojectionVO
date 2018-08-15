#include <iostream>
#include <cmath>
#include <cstdio>

#include "ceres/ceres.h"
#include "ceres/rotation.h"

#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>


namespace py = pybind11;

struct ReprojectionErrorResidual{

    ReprojectionErrorResidual(double* K, double* observed_pixel, double* point_cloud):
            _K(K), _observed_pixel(observed_pixel), _point_cloud(point_cloud){}


    template <typename T>
    bool operator()(const T* const transformation, T* residual) const {

        /* 1. Transforming point cloud with [R|t] */

        T transformed_point_cloud[3];

        T rotation[4];
        rotation[0] = transformation[6];    // qw
        rotation[1] = transformation[3];    // qx
        rotation[2] = transformation[4];    // qy
        rotation[3] = transformation[5];    // qz

        T translation[3];
        translation[0] = transformation[0]; // tx
        translation[1] = transformation[1]; // ty
        translation[2] = transformation[2]; // tz

        T tmp_point_cloud[3] = {T(_point_cloud[0]), T(_point_cloud[1]), T(_point_cloud[2])};

        // rotate 3d point cloud
        ceres::QuaternionRotatePoint(rotation, tmp_point_cloud, transformed_point_cloud);

//        std::cout << "point cloud: "
//                  << _point_cloud[0] <<
//                  "," << _point_cloud[1] <<
//                  "," << _point_cloud[2] << "\n";
//        std::cout << "after rotation: "
//                  << transformed_point_cloud[0] <<
//                  "," << transformed_point_cloud[1] <<
//                  "," << transformed_point_cloud[2] << "\n";

        // translate 3d point cloud after rotating
        transformed_point_cloud[0] += translation[0];
        transformed_point_cloud[1] += translation[1];
        transformed_point_cloud[2] += translation[2];

//        std::cout << "after translation: "
//                  << transformed_point_cloud[0] <<
//                  "," << transformed_point_cloud[1] <<
//                  "," << transformed_point_cloud[2] << "\n";


        /* 2. Convert 3d point cloud tp 2d pixel using camera intrinsics */

        T fx = T(_K[0]);
        T fy = T(_K[1]);
        T cx = T(_K[2]);
        T cy = T(_K[3]);

        T x = transformed_point_cloud[0];
        T y = transformed_point_cloud[1];
        T z = transformed_point_cloud[2];

        T u = fx * x / z + cx;
        T v = fy * y / z + cy;

//        std::cout << "converting to pixels: "
//                  << u << "," << v << "," << "\n";

        residual[0] = _observed_pixel[0] - u;
        residual[1] = _observed_pixel[1] - v;

//        std::cout << "residuals: "
//                  << residual[0] << "," << residual[1] << "," << "\n";

        return true;
    }

    static ceres::CostFunction* Create(double* K, double* observed_pixel, double* point_cloud) {
        return (new ceres::AutoDiffCostFunction<ReprojectionErrorResidual, 2, 7>(
                new ReprojectionErrorResidual(K, observed_pixel, point_cloud)));
    }


    double* _K;
    double* _observed_pixel;
    double* _point_cloud;

};

std::vector<double> calculate_residuals(
        py::array_t<double, py::array::c_style | py::array::forcecast> &transformation_list,
        py::array_t<double, py::array::c_style | py::array::forcecast> &observed_pixel_list,
        py::array_t<double, py::array::c_style | py::array::forcecast> &point_cloud_list){

    // check input dimensions
    if ( observed_pixel_list.ndim()     != 2 )
        throw std::runtime_error("observed_pixel_list should be 2-D NumPy array");
    if ( observed_pixel_list.shape()[1] != 2 )
        throw std::runtime_error("observed_pixel_list should have size [N,2]");

    if ( point_cloud_list.ndim()     != 2 )
        throw std::runtime_error("point_cloud_list should be 2-D NumPy array");
    if ( point_cloud_list.shape()[1] != 3 )
        throw std::runtime_error("point_cloud_list should have size [N,3]");


    if(observed_pixel_list.shape()[0] != point_cloud_list.shape()[0])
    {
        throw std::runtime_error("observed pixel's length and point cloud's size are not compatible");
    }

    int no_measurement = observed_pixel_list.shape()[0];

    double observed_pixel[no_measurement][2];

    std::memcpy(observed_pixel, observed_pixel_list.data(), observed_pixel_list.size()*sizeof(double));

    double point_cloud[no_measurement][3];
    std::memcpy(point_cloud, point_cloud_list.data(), point_cloud_list.size()*sizeof(double));

    // optimization variable => transformation => {tx,ty,tz,qx,qy,qz,qw}
    double transformation[7];
    std::memcpy(transformation, transformation_list.data(), transformation_list.size()*sizeof(double));


    // constant => intrinsic values: {fx, fy, cx, cy}
    double K[4] = {525.0, 525.0, 319.5, 239.5};


    ceres::Problem problem;

    for (int i = 0; i < no_measurement; i++) {
        ceres::CostFunction* cost_function = ReprojectionErrorResidual::Create(K, observed_pixel[i], point_cloud[i]);
        ceres::LossFunction* loss_function = new ceres::HuberLoss(5.991);
        problem.AddResidualBlock(cost_function, loss_function, transformation);
    }


    double cost = 0.0;
    std::vector<double> residuals;
    problem.Evaluate(ceres::Problem::EvaluateOptions(), &cost, &residuals, NULL, NULL);
    std::cout << "total cost: " << cost << std::endl;
//    std::cout << "residual vector: " << std::endl;
//    for (int i = 0; i < residuals.size(); i=i+2) {
//        std::cout << residuals[i] << "," << residuals[i+1] << std::endl;
//    }

    return residuals;
}


std::vector<double> optimize(
        py::array_t<double, py::array::c_style | py::array::forcecast> &transformation_list,
        py::array_t<double, py::array::c_style | py::array::forcecast> &observed_pixel_list,
        py::array_t<double, py::array::c_style | py::array::forcecast> &point_cloud_list){


    // check input dimensions
    if ( observed_pixel_list.ndim()     != 2 )
        throw std::runtime_error("observed_pixel_list should be 2-D NumPy array");
    if ( observed_pixel_list.shape()[1] != 2 )
        throw std::runtime_error("observed_pixel_list should have size [N,2]");

    if ( point_cloud_list.ndim()     != 2 )
        throw std::runtime_error("point_cloud_list should be 2-D NumPy array");
    if ( point_cloud_list.shape()[1] != 3 )
        throw std::runtime_error("point_cloud_list should have size [N,3]");


    if(observed_pixel_list.shape()[0] != point_cloud_list.shape()[0])
    {
        throw std::runtime_error("observed pixel's length and point cloud's size are not compatible");
    }

    int no_measurement = observed_pixel_list.shape()[0];

    double observed_pixel[no_measurement][2];

    std::memcpy(observed_pixel, observed_pixel_list.data(), observed_pixel_list.size()*sizeof(double));

    double point_cloud[no_measurement][3];
    std::memcpy(point_cloud, point_cloud_list.data(), point_cloud_list.size()*sizeof(double));

    // optimization variable => transformation => {tx,ty,tz,qx,qy,qz,qw}
    double transformation[7];
    std::memcpy(transformation, transformation_list.data(), transformation_list.size()*sizeof(double));


    // constant => intrinsic values: {fx, fy, cx, cy}
    double K[4] = {525.0, 525.0, 319.5, 239.5};


    ceres::Problem problem;

    for (int i = 0; i < no_measurement; i++) {
        ceres::CostFunction* cost_function = ReprojectionErrorResidual::Create(K, observed_pixel[i], point_cloud[i]);
        ceres::LossFunction* loss_function = new ceres::HuberLoss(5.991);
        problem.AddResidualBlock(cost_function, loss_function, transformation);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
//    options.max_num_iterations = 10;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
//    options.function_tolerance = 1.0e-10;
//    options.parameter_tolerance = 1.0e-10;
//    options.gradient_tolerance = 1.0e-10;
//    ceres::Solver::Summary summary;
//    ceres::Solve(options, &problem, &summary);
//    std::cout << summary.FullReport() << "\n";

    std::cout << "{tx,ty,tz,qx,qy,qz,qw}" << std::endl
              << transformation[0] <<
              "," << transformation[1] <<
              "," << transformation[2] <<
              "," << transformation[3] <<
              "," << transformation[4] <<
              "," << transformation[5] <<
              "," << transformation[6] << std::endl;



    std::vector<double> result(7);
    result[0] = transformation[0];
    result[1] = transformation[1];
    result[2] = transformation[2];
    result[3] = transformation[3];
    result[4] = transformation[4];
    result[5] = transformation[5];
    result[6] = transformation[6];

//    double R[9];
//    ceres::QuaternionToRotation(transformation, R);
//
//    std::cout << "optimized transformation: "
//              << R[0] <<
//              "," << R[1] <<
//              "," << R[2] << "\n" <<
//              "," << R[3] <<
//              "," << R[4] <<
//              "," << R[5] << "\n" <<
//              "," << R[6] <<
//              "," << R[7] <<
//              "," << R[8] << "\n";

//    ssize_t              ndim    = 2;
//    std::vector<ssize_t> shape   = { 1 , 7 };
//    std::vector<ssize_t> strides = { sizeof(double)*7 , sizeof(double) };
//
//    py::array result = py::array(py::buffer_info(
//            transformation,                           /* data as contiguous array  */
//            sizeof(double),                          /* size of one scalar        */
//            py::format_descriptor<double>::format(), /* data type                 */
//            ndim,                                    /* number of dimensions      */
//            shape,                                   /* shape of the matrix       */
//            strides                                  /* strides for each axis     */
//    ));

    return result;
}

PYBIND11_MODULE(ceres_reprojection, m)
{
    m.doc() = "pybind11 reprojection plugin"; // optional module docstring
    m.def("optimize", &optimize, "A function which optimize reprojection error");
    m.def("calculate_residuals", &calculate_residuals, "A function which calculates residual vector");
}


//void print_vector(const std::vector<int> &v) {
//    for (auto item : v)
//        std::cout << item << "\n";
//}

//void print_list(py::list my_list) {
//    for (auto item : my_list)
//        std::cout << item << " ";
//}

//int add(int i, int j) {
//    return i + j;
//}
//
//PYBIND11_MODULE(example, m) {
//    m.doc() = "pybind11 example plugin"; // optional module docstring
//
//    m.def("add", &add, "A function which adds two numbers");
//}
