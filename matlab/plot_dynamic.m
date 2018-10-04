function [y_plot,V_plot,error_plot,error_l_1,error_l_2,error_l_inf] = plot_dynamic(y_ref,y,lambda,V,config)

time_steps = size(y,2);
y_test = zeros(config.layers+1,config.n_ref);
lambda_test = y_test;
V_test = y_test;
for j=1:config.layers+1
    for i = 1:config.n_ref
        y_test(j,i) = y{j,i};
        lambda_test(j,i) = trace(lambda{j,i});
        V_test(j,i) = trace(V{j,i});
    end%for
end%for
MU_y = config.scaleshift.MU_y;
SIGMA_y  = config.scaleshift.SIGMA_y;
y_plot = undo_scaleshift(y_test(end,:),MU_y,SIGMA_y);
y_ref_plot = undo_scaleshift(y_ref,MU_y,SIGMA_y);
V_plot = undo_scaleshift(V_test(end,:),zeros(1,config.D_multi),SIGMA_y);
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

color_order = [0         0.4470    0.7410
                   0.8500    0.3250    0.0980
                   0.9290    0.6940    0.1250
                   0.4940    0.1840    0.5560
                   0.4660    0.6740    0.1880
                   0.3010    0.7450    0.9330
                   0.6350    0.0780    0.1840];
    color_order_var = [127 164 189
                       218 167 146
                       238 215 160
                       135 96  143
                       147 172 115
                       160 215 238
                       164 110 119]/255;
figure
for l = 1:config.layers 
    fill([1:time_steps fliplr(1:time_steps)],[y_test(l,:) + 2*sqrt(lambda_test(l,:)),fliplr(y_test(l,:) - 2*sqrt(lambda_test(l,:)))],color_order_var(l,:),'edgecolor',color_order_var(l,:),'DisplayName','2 times SD');
    hold on
end%for
for l = 1:config.layers 
    plot(1:time_steps,y_test(l,:),'DisplayName',strcat('state',num2str(l)),'color',color_order(l,:));
    hold on
end%for
title('predicted hidden states')
legend('show')

error_plot = y_plot  - y_ref_plot;

error_l_1 = l_1_error(y_plot,y_ref_plot,time_steps);
error_l_2 = l_2_error(y_plot,y_ref_plot,time_steps);
error_l_inf = l_inf_error(y_plot,y_ref_plot,time_steps);


end