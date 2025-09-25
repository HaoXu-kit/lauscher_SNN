class LearningRule:
    def grad_func(self, network_params, network_state, x, y_true):
        raise NotImplementedError

    def update_params(self, network_params, grads, learning_rate):
        raise NotImplementedError


def update_rule(learning_rate, path, p, g):
    param_name = path[-1].name
    if param_name in ["W_in", "W_rec"]:
        return p - learning_rate * g
    return p


class RTRLLearningRule(LearningRule):
    def grad_func(self, network_params, network_state, x, y_true):
        # TODO: Implement RTRL gradient calculation
        pass

    def update_params(self, network_params, grads, learning_rate):
        # TODO: Implement RTRL parameter update
        pass
