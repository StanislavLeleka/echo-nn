use crate::operation::param_operation::ParamOperation;
use ndarray::Array2;

pub struct WeightMultiply {
    pub param_operation: ParamOperation,
}

impl WeightMultiply {
    pub fn new(w: Array2<f64>) -> Self {
        Self {
            param_operation: ParamOperation::new(w),
        }
    }

    pub fn output(&self) -> Array2<f64> {
        self.param_operation
            .operation
            .input
            .as_ref()
            .unwrap()
            .dot(&self.param_operation.param)
    }

    pub fn input_grad(&self, output_grad: &Array2<f64>) -> Array2<f64> {
        output_grad.dot(&self.param_operation.param.t())
    }

    pub fn param_grad(&self, output_grad: &Array2<f64>) -> Array2<f64> {
        self.param_operation
            .operation
            .input
            .as_ref()
            .unwrap()
            .t()
            .dot(output_grad)
    }
}
