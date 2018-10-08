function [y_pred,V_pred,error_plot,error_l_1,error_l_2,error_l_inf] = deep_gp(X_exogen,X_exogen_ref,y,y_ref,config)
    error_l_2 = 'data for python created';
    error_l_1 = [];
    error_l_inf = [];
    error_plot = [];
    V_pred = [];
    y_pred = [];
    [B,opt_hyp]  = deep_gp_regression_train(X_exogen,y,config);
if strcmp(config.simulation,'on')
    config.hyp = opt_hyp;
    [y_test,lambda_test,V_test] = deep_gp_regression_test(X_exogen_ref,B,config);
    [y_pred,V_pred,error_plot,error_l_1,error_l_2,error_l_inf] = plot_dynamic(y_ref,y_test,lambda_test,V_test,config);
end %if

end