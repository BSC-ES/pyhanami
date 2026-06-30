import xarray as xr

from pyhanami.diags.scientific_evaluation import general, iso, mjo, tc
from pyhanami.utils.plots import plots_scientific_evaluation_tables, plots_tc


class ScientificEvaluationWrapper:
    """
    Wrapper class to handle multiple scientific evaluation instances for different
    simulation datasets.
    
    Parameters
    ----------
    evaluation_instances : list[XEvaluation]
        List of evaluation instances to handle.
    evaluation_type : str
        Type of evaluation to handle.
    """

    # Methods that should create combined plots
    COMBINED_MULTIMODEL_METHODS = {
        'scores_table',           # GeneralEvaluation + ISOEvaluation
        'ceof_corr_table',        # MJOEvaluation
        'ceof_bias_table',        # MJOEvaluation
        'mean_amplitude_bias_table',  # MJOEvaluation
        'active_days_bias_table',     # MJOEvaluation
        'activity_per_phase_bias_tables',  # MJOEvaluation
        'power_bias_table',       # MJOEvaluation
        'clim_bias_table',        # TCEvaluation
        'storm_bias_table',       # TCEvaluation
        'temp_corr_table',        # TCEvaluation
        'spatial_corr_table',     # TCEvaluation
        'linear_plots',           # TCEvaluation
    }


    def __init__(self, evaluation_instances, evaluation_type):

        # Map evaluation type to the corresponding class
        evaluation_classes = {
            "general": general.GeneralEvaluation,
            "iso": iso.ISOEvaluation,
            "mjo": mjo.MJOEvaluation,
            "tc": tc.TCEvaluation,
        }
        if evaluation_type not in evaluation_classes:
            raise ValueError(
                f"Invalid 'evaluation_type': '{evaluation_type}'. "
                f"Expected one of: {', '.join(evaluation_classes.keys())}."
            )

        # Validate input
        expected_type = evaluation_classes[evaluation_type]
        if isinstance(evaluation_instances, expected_type):
            evaluation_instances = [evaluation_instances]
        if not (
            isinstance(evaluation_instances, list)
            and all(isinstance(instance, expected_type) for instance in evaluation_instances)
        ):
            raise TypeError(
                f"'evaluation_instances' must be a list of {expected_type.__name__} instances "
                f"or a single {expected_type.__name__} instance."
            )
        self.instances = evaluation_instances
        self.evaluation_type = evaluation_type

        return


    def _combined_general_table(self, var_names=None, output_path=None, reference=True, **kwargs):
        """
        Generate and save/display a combined table plot with general scalar scores for the given
        dataset(s) and variable(s).

        Parameters
        ----------
        var_names : str or list[str], optional
            Climate variable(s) name(s). If None, all variables in the analysis will be used.
        output_path : str, optional
            Path to save the table plot. If None, the table is displayed but not saved.
        reference : bool
            Whether to display reference values in the first row (not colored) (default: True).
        """
        first_instance = self.instances[0]

        # Validate input
        if var_names is None:
            var_names = first_instance.var_names
        if isinstance(var_names, str):
            var_names = [var_names]
        for var_name in var_names:
            if var_name not in first_instance.var_names:
                raise ValueError(
                    f"Variable '{var_name}' was not used in the general scalar analysis. "
                    f"Available variables: {first_instance.var_names}"
                )

        # Prepare data and plotting parameters
        data = [instance.scores for instance in self.instances]
        data_names = [first_instance.obs_name] + [instance.sim_name for instance in self.instances]
        year_range = f"{first_instance.start_year}-{first_instance.end_year}"
        ensemble = first_instance.ensemble

        # Create and save/display plot
        plots_scientific_evaluation_tables.general_evaluation_scores_table(
            data=data,
            data_names=data_names,
            year_range=year_range,
            var_names=var_names,
            ensemble=ensemble,
            output_path=output_path,
            reference=reference,
            **kwargs
        )
        return


    def _combined_iso_table(self, output_path=None, reference=True, **kwargs):
        """
        Generate and save/display a combined table plot with ISO scalar scores for the given
        dataset(s).

        Parameters
        ----------
        output_path : str, optional
            Path to save the table plot. If None, the table is displayed but not saved.
        reference : bool
            Whether to display reference values in the first row (not colored) (default: True).
        """
        first_instance = self.instances[0]

        # Validate that scores are available
        if not all(instance.obs for instance in self.instances):
            raise ValueError(
                "Scalar scores cannot be computed without observations. "
                "Set `obs=True` when calling the `compute_iso_scores` method to compute them."
            )

        # Validate that all scores had the same treatment
        if not all(instance.correct_pc == first_instance.correct_pc for instance in self.instances):
            raise ValueError("All instances must have the same `correct_pc` treatment for combined ISO scores table.")
        correct_pc = first_instance.correct_pc


        # Prepare data and plotting parameters
        data = [instance.scores for instance in self.instances]
        data_names = [first_instance.obs_name] + [instance.sim_name for instance in self.instances]
        year_range = f"{first_instance.start_year_pc}-{first_instance.end_year_pc}"

        # Create and save/display plot
        plots_scientific_evaluation_tables.iso_evaluation_scores_table(
            data=data,
            data_names=data_names,
            year_range=year_range,
            correct_pc=correct_pc,
            output_path=output_path,
            reference=reference,
            **kwargs
        )

        return


    def _combined_mjo_table(self, method_name, output_path=None, reference=True, **kwargs):
        """
        Generate and save/display a combined table plot with MJO scalar scores for the given
        dataset(s).

        Parameters
        ----------
        method_name : str
            Name of the method to access from the Evaluation instances.
        output_path : str, optional
            Path to save the table plot. If None, the table is displayed but not saved.
        reference : bool
            Whether to display reference values in the first row (not colored) (default: True).
        """
        first_instance = self.instances[0]

        # Validate that all scores had the same treatment
        if not all(instance.data_res == first_instance.data_res for instance in self.instances):
            raise ValueError("All instances must have the same `data_res` for combined MJO scores table.")
        data_res = first_instance.data_res

        # Prepare data and plotting parameters
        data = None
        data_names = [first_instance.obs_name] + [instance.sim_name for instance in self.instances]
        year_range = f"{first_instance.start_year_mjo}-{first_instance.end_year_mjo}"


        # Create and save/display plot for each method
        if method_name == "ceof_corr_table":
            data = [instance.ceof_scores["ceof_corr"] for instance in self.instances]

        elif method_name == "ceof_bias_table":
            aux_datasets = []
            for instance in self.instances:
                aux_dataset = xr.Dataset({
                f"explained_var_bias_{m}": instance.ceof_scores["explained_var_bias"].sel(mode=m, drop=True)
                for m in range(2)
                })
                aux_dataset["max_lead_lag_corr_bias"] = instance.ceof_scores["max_lead_lag_corr_bias"]
                aux_dataset["pceof_bias"] = instance.ceof_scores["pceof_bias"]

                aux_datasets.append(aux_dataset)

            data = [ds.to_array() for ds in aux_datasets]

        elif method_name == "mean_amplitude_bias_table":
            data = [instance.activity_per_phase["mean_active_amplitude_bias"] for instance in self.instances]

        elif method_name == "active_days_bias_table":
            data = [instance.activity_per_phase["active_counts_bias"] for instance in self.instances]

        elif method_name == "power_bias_table":
            data = [
                instance.power_scores[["ew_ratio_bias", "eo_ratio_bias", "pwfps_bias"]].to_array()
                for instance in self.instances
            ]

        if data is not None:
            plots_scientific_evaluation_tables.mjo_evaluation_scores_table(
                data=data,
                data_names=data_names,
                year_range=year_range,
                method_name=method_name,
                data_res=data_res,
                output_path=output_path,
                reference=reference,
                **kwargs
            )
        elif method_name == "activity_per_phase_bias_tables":
            data_1 = [instance.activity_per_phase["mean_active_amplitude_bias"] for instance in self.instances]
            data_2 = [instance.activity_per_phase["active_counts_bias"] for instance in self.instances]

            plots_scientific_evaluation_tables.mjo_evaluation_scores_two_tables(
                data_1=data_1,
                data_2=data_2,
                data_names=data_names,
                year_range=year_range,
                output_path=output_path,
                reference=reference,
                **kwargs
            )
        else:
            raise ValueError(f"Method '{method_name}' is not recognized for MJO evaluation.")

        return


    def _combined_tc_table(self, method_name, output_path=None, reference=True, **kwargs):
        """
        Generate and save/display a combined table plot with TC bias/correlation scores for the given
        dataset(s).

        Parameters
        ----------
        method_name : str
            Name of the method to access from the Evaluation instances.
        output_path : str, optional
            Path to save the table plot. If None, the table is displayed but not saved.
        reference : bool
            Whether to display reference values in the first row (not colored) and reanalyses in 
            the subsequent rows (default: True).
        """
        first_instance = self.instances[0]

        # Validate that all scores had the same treatment
        if not all(instance.bin_size == first_instance.bin_size for instance in self.instances):
            raise ValueError("All instances must have the same `bin_size` for combined TC scores table.")
        bin_size = first_instance.bin_size

        # Prepare plotting parameters
        data_names = first_instance.model_names[:-1] + [instance.model_names[-1] for instance in self.instances]
        year_range = f"{first_instance.start_year_tc}-{first_instance.end_year_tc}"
        rows_to_remove = 0 if reference else len(first_instance.obs_names) + 1


        # Generate plots
        bias_type = None
        correlation_type = None

        if method_name == "clim_bias_table":
            bias_type = "climatological"
            data = [instance.clim_bias for instance in self.instances]
        elif method_name == "storm_bias_table":
            bias_type = "storm"
            data = [instance.storm_bias for instance in self.instances]
        elif method_name == "temp_corr_table":
            correlation_type = "seasonal"
            data = [instance.temp_corr for instance in self.instances]
        elif method_name == "spatial_corr_table":
            correlation_type = "spatial"
            data = [instance.spatial_corr for instance in self.instances]
        else:
            raise ValueError(f"Method '{method_name}' is not recognized for TC evaluation.")

        # Create and save/display plot
        common_kwargs = dict(
            data=data,
            data_names=data_names,
            year_range=year_range,
            bin_size=bin_size,
            output_path=output_path,
            rows_to_remove=rows_to_remove,
            **kwargs
        )

        if bias_type is not None:
            plots_scientific_evaluation_tables.tc_evaluation_bias_scores_table(
                **common_kwargs,
                bias_type=bias_type,
            )
        elif correlation_type is not None:
            plots_scientific_evaluation_tables.tc_evaluation_correlation_scores_table(
                **common_kwargs,
                correlation_type=correlation_type,
            )
        else:
            raise ValueError("Problem occurred when generating the TC scores table. "
                             "Please check the method name and parameters.")

        return


    def _combined_tc_linear(self, output_path=None, **kwargs):
        """
        Generate and save/display combined linear plots with monthly and interannual cycles of
        the TC metrics for the given dataset(s).

        Parameters
        ----------
        output_path : str, optional
            Path to save the table plot. If None, the table is displayed but not saved.
        """
        first_instance = self.instances[0]

        # Validate that all datasets had the same treatment
        if not all(instance.bin_size == first_instance.bin_size for instance in self.instances):
            raise ValueError("All instances must have the same `bin_size` for combined TC linear plots.")


        # Prepare data and plotting parameters
        data = [instance.data_cymep for instance in self.instances]
        data_names = first_instance.model_names[:-1] + [instance.model_names[-1] for instance in self.instances]
        year_init = first_instance.start_year_tc
        year_end = first_instance.end_year_tc

        # Create and save/display plot
        plots_tc.plot_linear_cycles(
            data=data,
            data_names=data_names,
            start_year=year_init,
            end_year=year_end,
            output_path=output_path,
            **kwargs
        )

        return


    def _combined_multimodel_plots(self, method_name, output_path=None, var_names=None, reference=True, **kwargs):
        """
        Generate and save/display combined plots for multiple evaluation instances based on the
        specified method name.

        Parameters
        ----------
        method_name : str
            Name of the method to access from the Evaluation instances.
        output_path : str, optional
            Path to save the plot. If None, the plot is displayed but not saved.
        var_names : str or list[str], optional
            Climate variable(s) name(s). If None, all variables in the analysis will be used.
        reference : bool
            Whether to display reference values in the first row (not colored) (default: True).
        """

        if self.evaluation_type == "general":
            self._combined_general_table(var_names, output_path, reference, **kwargs)

        elif self.evaluation_type == "iso":
            self._combined_iso_table(output_path, reference, **kwargs)

        elif self.evaluation_type == "mjo":
            self._combined_mjo_table(method_name, output_path, reference, **kwargs)

        elif self.evaluation_type == "tc":
            if method_name.endswith("_table"):
                self._combined_tc_table(method_name, output_path, reference, **kwargs)
            else:
                self._combined_tc_linear(output_path, **kwargs)

        return


    def __getattr__(self, method_name):
        """
        Access methods of the corresponding Evaluation instances.
        
        Parameters
        ----------
        method_name : str
            Name of the method to access from the Evaluation instances.

        Returns
        -------
        output : method
            Evaluated method.
        """

        # Check if the method exists in the first instance
        first_instance = self.instances[0]
        if not hasattr(first_instance, method_name):
            raise AttributeError(f"'{type(first_instance).__name__}' object has no attribute '{method_name}'.")

        # Check type of method using the first instance and apply to all instance if callable
        attr = getattr(first_instance, method_name)
        if len(self.instances) == 1 and method_name not in self.COMBINED_MULTIMODEL_METHODS:
            output = attr
        else:
            if callable(attr):
                if method_name in self.COMBINED_MULTIMODEL_METHODS:
                    # Treat combined plotting functions with specific methods
                    def combined_multimodel_wrapper(*args, **kwargs):
                        """Generate a single plot with data from all instances."""
                        return self._combined_multimodel_plots(method_name, *args, **kwargs)

                    output = combined_multimodel_wrapper
                else:
                    # For other callable methods, apply them directly to all instances
                    def method_wrapper(*args, **kwargs):
                        """Apply callable method to all instances and return list of results."""
                        results = [getattr(instance, method_name)(*args, **kwargs) for instance in self.instances]

                        # Do not return if all results are None (plotting/saving methods)
                        if all(r is None for r in results):
                            return None
                        return results

                    output = method_wrapper
            else:
                output = [getattr(instance, method_name) for instance in self.instances]

        return output


    def __getitem__(self, idx):
        """Allow indexing like a list."""
        return self.instances[idx]
