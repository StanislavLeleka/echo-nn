use crate::error::errors::ShapeError;
use crate::operation::operation::Operation;
use crate::utils::utils::assert_same_shape_v2;
use ndarray::Array2;

pub struct ParamOperation {
    pub operation: Operation,
    pub param: Array2<f64>,
    pub param_grad: Option<Array2<f64>>,
}

impl ParamOperation {
    pub fn new(param: Array2<f64>) -> Self {
        Self {
            operation: Operation::new(),
            param,
            param_grad: None,
        }
    }

    pub fn backward(&mut self, output_grad: &Array2<f64>) -> Result<&Array2<f64>, ShapeError> {
        assert_same_shape_v2(self.operation.output.as_ref().unwrap(), output_grad)?;

        self.operation.input_grad = Some(self.operation.input_grad(output_grad));
        self.param_grad = Some(self.param_grad(output_grad));

        assert_same_shape_v2(
            self.operation.input.as_ref().unwrap(),
            self.operation.input_grad.as_ref().unwrap(),
        )?;
        assert_same_shape_v2(&self.param, self.param_grad.as_ref().unwrap())?;

        Ok(self.operation.input_grad.as_ref().unwrap())
    }

    fn param_grad(&self, output_grad: &Array2<f64>) -> Array2<f64> {
        todo!()
    }
}
