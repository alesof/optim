.. Copyright (c) 2016-2023 Keith O'Hara

   Distributed under the terms of the Apache License, Version 2.0.

   The full license is in the file LICENSE, distributed with this software.

Sequential Unconstrained Minimization Technique
===============================================

**Table of contents**

.. contents:: :local:

----

Description
-----------

For a general problem

.. math::

   \min_x f(x) \text{ subject to } g_k (x) \leq 0, \ \ k \in \{1, \ldots, K \}

The Sequential Unconstrained Minimization Technique solves:

.. math::

   \min_x \left\{ f(x) + c(i) \times \frac{1}{2} \sum_{k=1}^K \left( \max \{ 0, g_k(x) \} \right)^2 \right\}

The algorithm stops when the error is less than ``err_tol``, or the total number of 'generations' exceeds a desired (or default) value.

----

Definitions
-----------

.. _sumt-func-ref1:
.. doxygenfunction:: sumt(ColVec_t& init_out_vals, std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data)> opt_objfn, void* opt_data, std::function<ColVec_t (const ColVec_t& vals_inp, Mat_t* jacob_out, void* constr_data)> constr_fn, void* constr_data)
   :project: optimlib

.. _sumt-func-ref2:
.. doxygenfunction:: sumt(ColVec_t& init_out_vals, std::function<fp_t (const ColVec_t& vals_inp, ColVec_t* grad_out, void* opt_data)> opt_objfn, void* opt_data, std::function<ColVec_t (const ColVec_t& vals_inp, Mat_t* jacob_out, void* constr_data)> constr_fn, void* constr_data, algo_settings_t& settings)
   :project: optimlib

----

Examples
--------

Rosenbrock Function
~~~~~~~~~~~~~~~~~~~

Code to run this example is given below. This is a simple example to minimize the `Rosenbrock function <https://en.wikipedia.org/wiki/Rosenbrock_function>`_, with nonlinear inequality constraints.

.. math::

   f(x,y) = 100(y - x^2)^2 + (1 - x)^2

subject to the constraints:

.. math::

   x^2 + y^2 \leq 2, \ x \leq 1, \ y \leq 1


.. toggle-header::
    :header: **Eigen (Click to show/hide)**

    .. code:: cpp

        #define OPTIM_ENABLE_EIGEN_WRAPPERS
        #include "optim.hpp"
        
        inline double rosenbrock_fn(const Eigen::VectorXd& vals_inp, Eigen::VectorXd* grad_out, void* opt_data)
         {
            double x = vals_inp(0);
            double y = vals_inp(1);

            double obj_val = 100*std::pow(y - x*x, 2) + std::pow(1 - x, 2);

            if(grad_out) {
               (*grad_out)(0) = -400*x*(y - x*x) + 2*x - 2;
               (*grad_out)(1) = 200*(y - x*x);
            }

            return obj_val;

         }

         Eigen::VectorXd inequality_cons(const Eigen::VectorXd& vals_inp, Eigen::MatrixXd* jacob_out, void* constr_data)
         {

            Eigen::VectorXd g(3);
            g(0) = 2 - vals_inp(0)*vals_inp(0) - vals_inp(1)*vals_inp(1);
            g(1) = 1 - vals_inp(0);
            g(2) = 1 - vals_inp(1);

            if(jacob_out) {
               jacob_out->resize(3, 2);
               (*jacob_out)(0, 0) = -2*vals_inp(0);
               (*jacob_out)(0, 1) = -2*vals_inp(1);
               (*jacob_out)(1, 0) = -1;
               (*jacob_out)(1, 1) = 0;
               (*jacob_out)(2, 0) = 0;
               (*jacob_out)(2, 1) = -1;
            }

            return g;

         }
        
        int main()
        {
            const int test_dim = 2;
            Eigen::VectorXd x = Eigen::VectorXd::Zero(test_dim);

            bool success = optim::sumt(x, rosenbrock_fn, nullptr, inequality_cons, nullptr);
        
            if (success) {
                std::cout << "sumt: Rosenbrock test completed successfully." << "\n";
            } else {
                std::cout << "sumt: Rosenbrock test completed unsuccessfully." << "\n";
            }
        
            std::cout << "sumt: solution to Rosenbrock test:\n" << x << std::endl;
        
            return 0;
        }
