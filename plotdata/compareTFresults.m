clear all
close all
clc

% H LH   RH   LE        RE        RS     LS       N 8x3 = 24
% 123 456 789 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
name = {'MKF, fixed', 'MKF, fixed + PCA','MKF + regularisation','MKF'};
figure;
for k = 1:4
    if (k==1)
        load ./MKF_fixed/camdata.txt
        load ./MKF_fixed/kinectdata.txt
    elseif (k==2)
        load ./MKF_PCA/camdata.txt
        load ./MKF_PCA//kinectdata.txt
    elseif (k==3)
        load ./MKF_Sampling_reg/camdata.txt
        load ./MKF_Sampling_reg/kinectdata.txt
    else
        load ./MKF/camdata.txt
        load ./MKF/kinectdata.txt
    end
    bins = 20:850;%[165:183 246:663 916:1149 1349:1829];
    % bins = [20:250 370:800];

    cam = zeros(length(bins),24);
    cam(:,1:3) = camdata(bins,4:6); %lh
    cam(:,4:6) = camdata(bins,10:12); %le
    cam(:,7:9) = camdata(bins,19:21); %ls
    cam(:,10:12) = camdata(bins,16:18); %rs
    cam(:,13:15) = camdata(bins,13:15); %re
    cam(:,16:18) = camdata(bins,7:9); %rh
    cam(:,19:21) = camdata(bins,22:24); %n
    cam(:,22:24) = camdata(bins,1:3); %h

    kinect{k} = zeros(length(bins),24);
    kinect{k}(:,1:3) = kinectdata(bins,7:9); %rh
    kinect{k}(:,4:6) =  kinectdata(bins,13:15); %re
    kinect{k}(:,7:9) = kinectdata(bins,16:18); %rs
    kinect{k}(:,10:12) = kinectdata(bins,19:21); %ls
    kinect{k}(:,13:15) = kinectdata(bins,10:12); %le
    kinect{k}(:,16:18) =  kinectdata(bins,4:6); %lh
    kinect{k}(:,19:21) = kinectdata(bins,22:24); %n
    kinect{k}(:,22:24) = kinectdata(bins,1:3); %h

    cols=['b','g','r','g','b','m'];

    % for j = 1:8
    %     err = sqrt((cam(:,3*j-2) - kinect(:,3*j-2)).^2 + (cam(:,3*j-1) - kinect(:,3*j-1)).^2 + (cam(:,3*j) - kinect(:,3*j)).^2);
    %     e(j,1) = mean(err);
    %     s(j,1) = std(err);
    % 
    % end

%     b = reshape(mean(cam(:,[7:12 19:24])),3,[])';
%     a = reshape(mean(kinect{k}(:,[7:12 19:24])),3,[])';
%     [d,z,transform] = procrustes(a,b,'Scaling',false,'Reflection',false);

    for j = 1:length(cam)
        b = reshape(cam(j,:),3,[])';
        a = reshape(kinect{1}(j,:),3,[])';
        [d,z,transform] = procrustes(a,b,'Scaling',false,'Reflection',false);
        for i = 1:8
            cam1{k}(j,3*i-2:3*i) = cam(j,3*i-2:3*i)*transform.T + transform.c(1,:);
        end
    end

    subplot(4,1,k)
    hold on
    cc=hsv(8);
    for j = 1:8
        err{k} = 1000*sqrt((cam1{k}(:,3*j-2) - kinect{1}(:,3*j-2)).^2 + (cam1{k}(:,3*j-1) - kinect{1}(:,3*j-1)).^2 + (cam1{k}(:,3*j) - kinect{1}(:,3*j)).^2);
        plot(err{k},'color',cc(j,:),'LineWidth',2);
        e(j,k) = mean(err{k});
        s(j,k) = std(err{k});

    end
    hold off;
    title(name{k})
    ylabel('Tracking error (mm)')
    xlabel('Sample')
    axis([1 850,0 500]);
    grid on
end
legend('Left hand','Left elbow','Left shoulder','Right shoulder','Right elbow','Right hand','Neck','Head')

figure;
barwitherr(s,e)
set(gca,'XTickLabel',{'Right hand', 'Right elbow', 'Right shoulder', 'Left shoulder', 'Left elbow','Left Hand','Neck','Head'})
ylabel('Average tracking error (mm)')
legend(name)

figure;
T = 10;
col = {'b','g','c','m'};
for j = (T+1):length(cam1{4})
    subplot(1,2,1)
    cla;
    for k = 1:4
        hold on;
        for i = 1:5
            line([cam1{k}(j,3*i-2) cam1{k}(j,3*(i+1)-2)],[cam1{k}(j,3*i-1) cam1{k}(j,3*(i+1)-1)],[cam1{k}(j,3*i) cam1{k}(j,3*(i+1))],'Color',col{k},'LineWidth',4);
            line([kinect{1}(j,3*i-2) kinect{1}(j,3*(i+1)-2)],[kinect{1}(j,3*i-1) kinect{1}(j,3*(i+1)-1)],[kinect{1}(j,3*i) kinect{1}(j,3*(i+1))],'Color','r','LineWidth',4);
        end
        line([cam1{k}(j,3*7-2) cam1{k}(j,3*(8)-2)],[cam1{k}(j,3*7-1) cam1{k}(j,3*(8)-1)],[cam1{k}(j,3*7) cam1{k}(j,3*(8))],'Color',col{k},'LineWidth',4);
        line([kinect{1}(j,3*7-2) kinect{1}(j,3*(8)-2)],[kinect{1}(j,3*7-1) kinect{1}(j,3*(8)-1)],[kinect{1}(j,3*7) kinect{1}(j,3*(8))],'Color','r','LineWidth',4);
        hold off;
        grid on;
        axis([-0.6 0.6,-0.8 0.2,-0.8 0.6])
    end
    subplot(1,2,2)
    cla;
    hold all
    for k = 1:3
        plot(err{k}(j-T:j),'color',col{k})
    end
    axis([1 10,0 500])
    pause(0.01)
end




