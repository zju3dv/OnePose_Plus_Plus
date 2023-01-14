from time import time
import torch
import torch.optim as optim


def FirstOrderSolve(
    variables,
    constants,
    indices,
    fn,
    optimization_cfgs=None,
    verbose=False,
    return_residual_inform=False,
):
    """
    Parameters:
    ------------
    tragetory_dict: Dict{'w_kpts0_list':[[torch.tensor L*2][]]}
    """
    # Make confidance:
    initial_confidance = 50
    confidance = torch.full(
        (constants[0].shape[0],), fill_value=initial_confidance, requires_grad=False
    )
    max_steps = optimization_cfgs["max_steps"]

    # variable, constants and optimizer initialization
    variables = [torch.nn.Parameter(v) for v in variables]

    optimizer_type = optimization_cfgs["optimizer"]
    if "lr" in optimization_cfgs:
        # pose and depth refinement scenario
        lr = optimization_cfgs["lr"]
        # optimizer = optim.AdamW(variables, lr=lr)
        if optimizer_type == "Adam":
            optimizer = optim.Adam(variables, lr=lr)
        elif optimizer_type in ["SGD", "RMSprop"]:
            if optimizer_type == "SGD":
                optimizerBuilder = optim.SGD
            elif optimizer_type == "RMSprop":
                optimizerBuilder = optim.RMSprop
            else:
                raise NotImplementedError
            optimizer = optimizerBuilder(
                variables,
                lr=lr,
                momentum=optimization_cfgs["momentum"],
                weight_decay=optimization_cfgs["weight_decay"],
            )
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    constantsPar = constants

    start_time = time()
    for i in range(max_steps):
        current_iteration_start_time = time()

        variables_expanded = []
        if isinstance(indices, list):
            # indices: [indexs: n]
            for variable_id, variable_index in enumerate(indices):
                variable_expanded = variables[variable_id][variable_index]
                variables_expanded.append(variable_expanded)
        elif isinstance(indices, dict):
            # indices: {variable_id: [indexs: n]}
            for variable_id, variable_indexs in indices.items():
                assert isinstance(variable_indexs, list)
                variable_expanded = [
                    variables[variable_id][variable_index]
                    for variable_index in variable_indexs
                ]

                if variable_id == 0:
                    variable_expanded[0] = variable_expanded[
                        0
                    ].detach()  # Left Pose Scenario

                variables_expanded += variable_expanded
        else:
            raise NotImplementedError

        optimizer.zero_grad()
        try:
            results, confidance = fn(
                *variables_expanded,
                *constantsPar,
                confidance=confidance,
                verbose=verbose,
                marker=False,
                marker_return=True
            )
        except:
            results = fn(
                *variables_expanded,
                *constantsPar,
                confidance=confidance,
                verbose=verbose,
                marker=False,
                marker_return=True
            )
        if isinstance(results, torch.Tensor):
            residuals = results
        else:
            residuals, _ = results

        l = torch.sum(0.5 * residuals * residuals)
        l.backward()
        optimizer.step()

        current_step_residual = l.clone().detach()
        current_time = time()
        if i == 0:
            initial_residual = current_step_residual
            last_residual = initial_residual
            print(
                "Start one order optimization, residual = %E, total_time = %f ms"
                % (initial_residual, (current_time - start_time) * 1000)
            ) if verbose else None

        else:
            relative_decrease_rate = (
                last_residual - current_step_residual
            ) / last_residual
            print(
                "iter = %d, residual = %E, relative decrease percent= %f%%, current_iter_time = %f ms, total time = %f ms, %d residuals filtered"
                % (
                    i - 1,
                    current_step_residual,
                    relative_decrease_rate * 100,
                    (current_time - current_iteration_start_time) * 1000,
                    (current_time - start_time) * 1000,
                    torch.sum(confidance <= 0) if confidance is not None else 0,
                )
            ) if verbose else None
            last_residual = current_step_residual
            if relative_decrease_rate < 0.0001 and i > max_steps * 0.2:
                print("early stop!") if verbose else None
                break

    start_time = time()
    with torch.no_grad():
        results, confidance = fn(
            *variables_expanded,
            *constantsPar,
            confidance=confidance,
            verbose=verbose,
            marker=False,
            marker_return=True
        )
        if isinstance(results, torch.Tensor):
            residuals = results
        else:
            residuals, _ = results
        finial_residual = torch.sum(0.5 * residuals * residuals)
    print(
        "First order optimizer initial residual = %E , finial residual = %E, decrease = %E, relative decrease percent = %f%%, %d residuals filtered"
        % (
            initial_residual,
            finial_residual,
            initial_residual - finial_residual,
            ((initial_residual - finial_residual) / (initial_residual + 1e-4) * 100),
            torch.sum(confidance <= 0) if confidance is not None else 0,
        )
    ) if verbose else None
    variables = [variable.detach() for variable in variables]

    if return_residual_inform:
        return variables, [initial_residual, finial_residual]
    else:
        return variables
