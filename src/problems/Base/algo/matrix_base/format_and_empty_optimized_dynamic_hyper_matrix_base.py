from src.matrix.format_optimized_matrix import FormatOptimizedMatrix
from src.matrix.enhanced_matrix import EnhancedMatrix
from src.matrix.short_circuiting_for_empty_matrix import ShortCircuitingForEmptyMatrix
from src.problems.Base.algo.matrix_base.abstract_dynamic_hyper_matrix_base import AbstractDynamicHyperMatrixBaseAlgo


class FormatAndEmptyOptimizedDynamicHyperMatrixBaseAlgo(AbstractDynamicHyperMatrixBaseAlgo):
    def non_hyper_enhance_matrix(self, base_matrix: EnhancedMatrix, var_name: str) -> EnhancedMatrix:
        return FormatOptimizedMatrix(
            # base_matrix is CSR, and we don't need to cache CSR if var_name never occurs in place of r2 in rules
            discard_base_on_reformat=(
                        var_name not in self.r2s_with_hyper_r1 and var_name not in self.r2s_with_non_hyper_r1),
            base=ShortCircuitingForEmptyMatrix(base_matrix),
        )
