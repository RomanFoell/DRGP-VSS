function [y_plot,V_plot,error_plot,error_l_1,error_l_2,error_l_inf] = plot_dynamic(y_ref,y,V,config)

time_steps = size(y_ref,2);
y_test = zeros(config.D_multi,config.n_ref);
V_test = y_test;
for i = 1:config.n_ref
    y_test(:,i) = y{end,i};
    V_test(:,i) = trace(V{end,i});
end%for
MU_y = config.scaleshift.MU_y;
SIGMA_y  = config.scaleshift.SIGMA_y;
y_plot = undo_scaleshift(y_test,MU_y,SIGMA_y);
y_ref_plot = undo_scaleshift(y_ref,MU_y,SIGMA_y);
V_plot = undo_scaleshift(V_test,zeros(1,config.D_multi),SIGMA_y);
% scaled RMSE/plot
% y_plot = y_test;
% y_ref_plot = y_ref;
% V_plot = V_test;
figure
fill([1:time_steps fliplr(1:time_steps)],[y_plot + 2*sqrt(V_plot),fliplr(y_plot - 2*sqrt(V_plot))],[0.7, 0.7, 0.7],'edgecolor',[0.7, 0.7, 0.7],'DisplayName','2 times SD');
hold on
plot(1:time_steps,y_ref_plot,'b','DisplayName','real data');
hold on
plot(1:time_steps,y_plot,'k','DisplayName','simulation');

legend('show')

error_plot = y_plot  - y_ref_plot;

error_l_1 = l_1_error(y_plot,y_ref_plot,time_steps);
error_l_2 = l_2_error(y_plot,y_ref_plot,time_steps);
error_l_inf = l_inf_error(y_plot,y_ref_plot,time_steps);


end