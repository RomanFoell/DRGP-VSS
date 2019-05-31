
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DYNAMIC DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if strcmp(data_set,'actuator')

%%%%%%%%%%%%%%%%%%%%%%%%%
% data from mattos article, actuator
%%%%%%%%%%%%%%%%%%%%%%%%%

cd data_raw
load actuator.mat
cd ..
X_ref = X(:,513:end);
y_ref = y(513:end);
X = X(:,1:512);
y = y(1:512);

elseif strcmp(data_set,'drive')

%%%%%%%%%%%%%%%%%%%%%%%%%
% data from mattos article, drive
%%%%%%%%%%%%%%%%%%%%%%%%%

cd data_raw
load drive.mat
cd ..
X_ref = u1(251:end)';
y_ref = z1(251:end)';
X = u1(1:250)';
y = z1(1:250)';

elseif strcmp(data_set,'damper')

%%%%%%%%%%%%%%%%%%%%%%%%%
% data from svensson article, damper
%%%%%%%%%%%%%%%%%%%%%%%%%

cd data_raw
load damper.mat
cd ..
X_ref = X(:,2001:end);
y_ref = y(2001:end);
X = X(:,1:2000);
y = y(1:2000);

elseif strcmp(data_set,'emission')

%%%%%%%%%%%%%%%%%%%%%%%%%
% data from DOE Tagung, emission
% %%%%%%%%%%%%%%%%%%%%%%%%%

cd data_raw
load emission.mat
cd ..
X_ref = X(:,10001:end);
y_ref = y(:,10001:end);
X = X(:,1:10000);
y = y(:,1:10000);

elseif strcmp(data_set,'power_load')

% %%%%%%%%%%%%%%%%%%%%%%%%%
% % data from svensson article, power load
% %%%%%%%%%%%%%%%%%%%%%%%%%

cd data_raw
load power_load.mat
cd ..
X_ref = X(:,28556:4:end);
y_ref = y(:,28556:4:end);
X = X(:,1:4:28553);
y = y(:,1:4:28553);

elseif strcmp(data_set,'ballbeam')

%%%%%%%%%%%%%%%%%%%%%%%%%
% data from svensson article, ballbeam
%%%%%%%%%%%%%%%%%%%%%%%%%

cd data_raw
load ballbeam.mat
cd ..
X_ref = X(:,501:end);
y_ref = y(:,501:end);
X = X(:,1:500);
y = y(:,1:500);

elseif strcmp(data_set,'dryer')

%%%%%%%%%%%%%%%%%%%%%%%%%
% data from svensson article, dryer
%%%%%%%%%%%%%%%%%%%%%%%%%

cd data_raw
load dryer.mat
cd ..
X_ref = X(:,501:end);
y_ref = y(:,501:end);
X = X(:,1:500);
y = y(:,1:500);

elseif strcmp(data_set,'cascaded')
%%%%%%%%%%%%%%%%%%%%%%%%%
% data from mattos dissertation, cascaded
%%%%%%%%%%%%%%%%%%%%%%%%%

cd data_raw
load cascaded.mat
cd ..
X_ref = X(:,1025:end);
y_ref = y(:,1025:end);
X = X(:,1:1024);
y = y(:,1:1024);

elseif strcmp(data_set,'sarco')
%%%%%%%%%%%%%%%%%%%%%%%%%
% data sarco, rasmussen
%%%%%%%%%%%%%%%%%%%%%%%%%
    
cd data_raw
load sarcos_inv.mat
cd ..
% X = sarcos_inv(26000:28000,22:end)';
X = [sarcos_inv(26000:28000,22:end)';sarcos_inv(26000:28000,1)'];
y = sarcos_inv(26000:28000,1)';
cd data_raw
load sarcos_inv_test.mat
cd ..
% X_ref = sarcos_inv_test(2600:3500,22:end)';
X_ref = [sarcos_inv_test(2600:3500,22:end)';sarcos_inv_test(2599:3499,1)'];
y_ref = sarcos_inv_test(2600:3500,1)';

end%if

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
