use crate::error::errors::ShapeError;
use crate::operation::param_operation::ParamOperation;
use ndarray::Array2;

pub struct BiasAdd {
    pub param_operation: ParamOperation,
}

impl BiasAdd {
    pub fn new(b: Array2<f64>) -> Result<Self, ShapeError> {
        if b.shape()[0] != 1 {
            return Err(ShapeError {
                expected: vec![1, b.shape()[1]],
                found: b.shape().to_vec(),
            });
        }

        Ok(BiasAdd {
            param_operation: ParamOperation::new(b),
        })
    }

    pub fn output(&self) -> Array2<f64> {
        self.param_operation.operation.input.as_ref().unwrap() + &self.param_operation.param
    }

    pub fn input_grad(&self, output_grad: &Array2<f64>) -> Array2<f64> {
        Array2::ones(
            self.param_operation
                .operation
                .input
                .as_ref()
                .unwrap()
                .raw_dim(),
        ) * output_grad
    }

    pub fn param_grad(&self, output_grad: &Array2<f64>) -> Array2<f64> {
        let param_grad = Array2::ones(self.param_operation.param.raw_dim()) * output_grad;
        let summed = param_grad.sum_axis(ndarray::Axis(0));
        summed.into_shape((1, param_grad.shape()[1])).unwrap()
    }
}
