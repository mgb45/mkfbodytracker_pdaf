clear all
close all
clc

% H LH   RH   LE        RE        RS     LS       N 8x3 = 24
% 123 456 789 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
clusters = [1 3 6 11 17 22 30 26 22 23 21 24 29 37 37 47 42 36 40 38];
% name = {'Baseline',  'd = 2', 'd = 4', 'd = 6','d = 8', 'd = 10', 'd = 12', 'd = 14', 'd = 16', 'd = 18', 'd = 20'};

dat_path = './';
files = dir(strcat(dat_path,'*.txt'));
K = length(files)/2;
kinect = cell(K,1);
cam = cell(K,1);
k = 0;
j = 0;
for n = 1:length(files)
    bins = 20:850;%[165:183 246:663 916:1149 1349:1829];
    
    if strcmp(files(n).name(1:6),'kinect')
        kinectdata = load(strcat(dat_path,files(n).name));
        k = k + 1;
        kinect{k} = zeros(length(bins),24);
        kinect{k}(:,1:3) = kinectdata(bins,7:9); %rh
        kinect{k}(:,4:6) =  kinectdata(bins,13:15); %re
        kinect{k}(:,7:9) = kinectdata(bins,16:18); %rs
        kinect{k}(:,10:12) = kinectdata(bins,19:21); %ls
        kinect{k}(:,13:15) = kinectdata(bins,10:12); %le
        kinect{k}(:,16:18) =  kinectdata(bins,4:6); %lh
        kinect{k}(:,19:21) = kinectdata(bins,22:24); %n
        kinect{k}(:,22:24) = kinectdata(bins,1:3); %h
    else
        j = j+1;
        camdata = load(strcat(dat_path,files(n).name));
        cam{j} = zeros(length(bins),24);
        cam{j}(:,1:3) = camdata(bins,4:6); %lh
        cam{j}(:,4:6) = camdata(bins,10:12); %le
        cam{j}(:,7:9) = camdata(bins,19:21); %ls
        cam{j}(:,10:12) = camdata(bins,16:18); %rs
        cam{j}(:,13:15) = camdata(bins,13:15); %re
        cam{j}(:,16:18) = camdata(bins,7:9); %rh
        cam{j}(:,19:21) = camdata(bins,22:24); %n
        cam{j}(:,22:24) = camdata(bins,1:3); %h
    end
end

for k = 1:K
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

    for j = 1:length(cam{k})
        b = reshape(cam{k}(j,:),3,[])';
        a = reshape(kinect{1}(j,:),3,[])';
        [d,z,transform] = procrustes(a,b,'Scaling',false,'Reflection',false);
        for i = 1:8
            cam1{k}(j,3*i-2:3*i) = cam{k}(j,3*i-2:3*i)*transform.T + transform.c(1,:);
        end
    end

    subplot(ceil(K/3),3,k)
    hold on
    cc=hsv(8);
    for j = 1:8
        err{k}(:,j) = 1000*sqrt((cam1{k}(:,3*j-2) - kinect{1}(:,3*j-2)).^2 + (cam1{k}(:,3*j-1) - kinect{1}(:,3*j-1)).^2 + (cam1{k}(:,3*j) - kinect{1}(:,3*j)).^2);
        plot(err{k}(:,j),'color',cc(j,:),'LineWidth',2);
        e(j,k) = mean(err{k}(:,j));
        s(j,k) = std(err{k}(:,j));

    end
    hold off;
%     title(name{k})
    ylabel('Tracking error (mm)')
    xlabel('Sample')
    axis([1 850,0 600]);
    grid on
end
legend('Left hand','Left elbow','Left shoulder','Right shoulder','Right elbow','Right hand','Neck','Head')

figure;
barwitherr(s,e)
set(gca,'XTickLabel',{'Left hand','Left elbow','Left shoulder','Right shoulder','Right elbow','Right hand','Neck','Head'})
ylabel('Average tracking error (mm)')
% legend(name)

figure;
hold on;
N = histc(mean(err{1},2),0:0.01:200);
A{1} = plot(0:0.01:200,cumsum(N)./sum(N),'k','LineWidth',4);
pcp_m = zeros(K-1,length(N));
for j = 2:K
    N = histc(mean(err{j},2),0:0.01:200);
    pcp_m(j-1,:) = cumsum(N)./sum(N);
end
A{2} = shadedErrorBar(0:0.01:200,mean(pcp_m),std(pcp_m),{'-b','LineWidth',2});
legend([A{1},A{2}.mainLine],'Baseline','Rao-Blackwelised + ML (k=27, d=21)')
grid on
ylabel('Classification rate')
xlabel('Average joint error threshold (mm)')

% figure;
% T = 10;
% col = hsv(K);
% for j = (T+1):length(cam1{4})
%     set(0,'CurrentFigure',4)
%     subplot(1,2,1)
%     cla;
%     for k = 1:K
%         hold on;
%         for i = 1:5
%             line([cam1{k}(j,3*i-2) cam1{k}(j,3*(i+1)-2)],[cam1{k}(j,3*i-1) cam1{k}(j,3*(i+1)-1)],[cam1{k}(j,3*i) cam1{k}(j,3*(i+1))],'Color',col(k,:),'LineWidth',2);
%             line([kinect{1}(j,3*i-2) kinect{1}(j,3*(i+1)-2)],[kinect{1}(j,3*i-1) kinect{1}(j,3*(i+1)-1)],[kinect{1}(j,3*i) kinect{1}(j,3*(i+1))],'Color','k','LineWidth',4);
%         end
%         line([cam1{k}(j,3*7-2) cam1{k}(j,3*(8)-2)],[cam1{k}(j,3*7-1) cam1{k}(j,3*(8)-1)],[cam1{k}(j,3*7) cam1{k}(j,3*(8))],'Color',col(k,:),'LineWidth',2);
%         line([kinect{1}(j,3*7-2) kinect{1}(j,3*(8)-2)],[kinect{1}(j,3*7-1) kinect{1}(j,3*(8)-1)],[kinect{1}(j,3*7) kinect{1}(j,3*(8))],'Color','k','LineWidth',4);
%         hold off;
%         grid on;
%         axis([-0.6 0.6,-0.8 0.2,-0.8 0.6])
%     end
%     subplot(1,2,2)
%     cla;
%     hold all
%     for k = 1:K
%         plot(err{k}(j-T:j),'color',col(k,:))
%     end
%     legend(name)
%     axis([1 10,0 500])
%     grid on
%     pause(0.01)
% end