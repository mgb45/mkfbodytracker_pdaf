close all
clear all
clc;

load('WeightsL3D.txt')
load ('MeansL3D.txt');
load ('CovsL3D.txt');

Ns = 1000;
N = 5e3; % number samples

d = 21; % State dimensionality
col = {'-b','-r','-g','-c','-m','-k','-y'};

err = zeros(Ns,N,5);
for ns = 1:Ns

    g = 0.8*rand(1);
    k = randi(30);
    mu = MeansL3D(k,:);

    si = CovsL3D(d*k-d+1:d*k,:);
    Q = (1-g^2)*si;
    Qchol = chol(Q);

    % Simulate process
    x = zeros(N,d); 
    gEst = zeros(N,3); 

    x(1,:) = mu + randn(1,d)*Qchol; 
    C = zeros(2*d);
    u = 0; v =0;
    d2 = d*2;
    for j = 1:N 
        x(j+1,:) = mu + g*(x(j,:)-mu) + randn(1,d)*Qchol; 
        y = [x(j,:) x(j+1,:)] - [mu mu];
        C = C + y'*y;
        u = u + y(1:d)/si*y(1:d)' + y(d+1:end)/si*y(d+1:end)';
        v = v + y(1:d)/si*y(d+1:end)' + y(d+1:end)/si*y(1:d)';

        gEst(j,1) = sum(diag(C/j,d))./sum(diag(si));
        gEst(j,2) = sqrt(1-(det(C/j)/det(si)^2)^(1/d));
        gEst(j,3) = sum(sum(C(d+1:end,1:d)/j.*si))/sum(si(:).*si(:));
        gEst(j,4) = sum(sum(C(d+1:end,1:d)/j.*C(1:d,1:d)/j))/sum(sum(C(1:d,1:d)/j.*C(1:d,1:d)/j));

        % Routh hurwitz number roots
        ca = j*d2; cb = -v; cc = (2*u-j*d2); cd = -v;
  
        del0 = cb^2 - 3*ca*cc;
        del1 = 2*cb^3 - 9*ca*cb*cc + 27*ca^2*cd;

        u1 = 1;
        u2 = (-1+1i*sqrt(3))/2;
        u3 = (-1-1i*sqrt(3))/2;
        
        Cu = ((del1 + sqrt(del1^2 - 4*del0^3))/2)^(1/3);

        gEst(j,5) = -1/(3*ca)*(cb + u1*Cu + del0/(u1*Cu));
%         gEst(j,6) = -1/(3*ca)*(cb + u2*Cu + del0/(u2*Cu));
%         gEst(j,7) = -1/(3*ca)*(cb + u3*Cu + del0/(u3*Cu));

%         if (mod(j,1000) == 0) 
%             plot((1:j)/30,repmat(g,j,1))
%             hold all;
%             for k = 1:5
%                 plot((1:j)/30,gEst(1:j,k))
%             %     plot((1:j)/30,abs(gML(1:j,k)))
%             end
%             hold off;
%             grid on;
%             % axis([1/30 j/30, 0 1])
%             ylabel('\gamma')
%             xlabel('Time (s)')
%             legend('Target','Mean of diag','Determinant','Least squares using Si','Least squares','ML1','ML2','ML3')
%             drawnow;
%     
%         end

    end

    err(ns,:,:) = real(gEst - g);

    if (ns > 2)
        hold off;
        for j = 1:5
            A{j} = shadedErrorBar([],mean(err(1:ns,:,j),1)',std(err(1:ns,:,j),1)',col{j},1);
            hold on;
        end
        grid on;
        ylabel('Error')
        xlabel('Samples')
        % axis([1 141,-17 450])
        B = cell2mat(A);
        legend([B.mainLine],'Mean of diag','Determinant','Least squares using Si','Least squares','ML')%,'ML2','ML3')
        drawnow;
    end

end

% C = C/N;
% 
% plot((1:j)/30,repmat(g,j,1))
% hold all;
% % for k = 1:4
% %     plot((1:j)/30,gEst(1:j,k))
% % %     plot((1:j)/30,abs(gML(1:j,k)))
% % end
% for k =1:3
%     plot((1:j)/30,abs(gML(1:j,k)))
% end
% hold off;
% grid on;
% % axis([1/30 j/30, 0 1])
% ylabel('\gamma')
% xlabel('Time (s)')
% % legend('Target','Mean of diag','Determinant','Least squares using Si','Least squares')%,'ML')
% drawnow;

% var(gEst)
% 
% figure;
% subplot(1,2,1)
% imagesc(C/N)
% subplot(1,2,2)
% imagesc([si g*si; g*si si])


% N*d*2*gML.^3 +gML.^2*v -(N*2*d+2*u)*gML + v

% gML(end,:)
% g

% gs = (C(22:end,1:21)./si);
% [median(gs(:)) gEst(end) g sqrt(1-(det(C)/det(si)^2)^(1/d))]
% gest = C(1,2)/si
% gest1 = C(1,2)/mean(diag(C))




%     gML(j,1) = (((v/(4*j*d) + v^3/(216*j^3*d^3) + (v*(2*u + 2*j*d))/(24*j^2*d^2))^2 - ((2*u + 2*j*d)/(6*j*d) + v^2/(36*j^2*d^2))^3)^(1/2) - v/(4*j*d) - v^3/(216*j^3*d^3) - (v*(2*u + 2*j*d))/(24*j^2*d^2))^(1/3) + ((2*u + 2*j*d)/(6*j*d) + v^2/(36*j^2*d^2))/(((v/(4*j*d) + v^3/(216*j^3*d^3) + (v*(2*u + 2*j*d))/(24*j^2*d^2))^2 - ((2*u + 2*j*d)/(6*j*d) + v^2/(36*j^2*d^2))^3)^(1/2) - v/(4*j*d) - v^3/(216*j^3*d^3) - (v*(2*u + 2*j*d))/(24*j^2*d^2))^(1/3) - v/(6*j*d);
%     gML(j,2) = -(((v/(4*j*d) + v^3/(216*j^3*d^3) + (v*(2*u + 2*j*d))/(24*j^2*d^2))^2 - ((2*u + 2*j*d)/(6*j*d) + v^2/(36*j^2*d^2))^3)^(1/2) - v/(4*j*d) - v^3/(216*j^3*d^3) - (v*(2*u + 2*j*d))/(24*j^2*d^2))^(1/3)/2 - ((2*u + 2*j*d)/(6*j*d) + v^2/(36*j^2*d^2))/(2*(((v/(4*j*d) + v^3/(216*j^3*d^3) + (v*(2*u + 2*j*d))/(24*j^2*d^2))^2 - ((2*u + 2*j*d)/(6*j*d) + v^2/(36*j^2*d^2))^3)^(1/2) - v/(4*j*d) - v^3/(216*j^3*d^3) - (v*(2*u + 2*j*d))/(24*j^2*d^2))^(1/3)) - (3^(1/2)*((((v/(4*j*d) + v^3/(216*j^3*d^3) + (v*(2*u + 2*j*d))/(24*j^2*d^2))^2 - ((2*u + 2*j*d)/(6*j*d) + v^2/(36*j^2*d^2))^3)^(1/2) - v/(4*j*d) - v^3/(216*j^3*d^3) - (v*(2*u + 2*j*d))/(24*j^2*d^2))^(1/3) - ((2*u + 2*j*d)/(6*j*d) + v^2/(36*j^2*d^2))/(((v/(4*j*d) + v^3/(216*j^3*d^3) + (v*(2*u + 2*j*d))/(24*j^2*d^2))^2 - ((2*u + 2*j*d)/(6*j*d) + v^2/(36*j^2*d^2))^3)^(1/2) - v/(4*j*d) - v^3/(216*j^3*d^3) - (v*(2*u + 2*j*d))/(24*j^2*d^2))^(1/3))*i)/2 - v/(6*j*d);
%     gML(j,3) = -(((v/(4*j*d) + v^3/(216*j^3*d^3) + (v*(2*u + 2*j*d))/(24*j^2*d^2))^2 - ((2*u + 2*j*d)/(6*j*d) + v^2/(36*j^2*d^2))^3)^(1/2) - v/(4*j*d) - v^3/(216*j^3*d^3) - (v*(2*u + 2*j*d))/(24*j^2*d^2))^(1/3)/2 - ((2*u + 2*j*d)/(6*j*d) + v^2/(36*j^2*d^2))/(2*(((v/(4*j*d) + v^3/(216*j^3*d^3) + (v*(2*u + 2*j*d))/(24*j^2*d^2))^2 - ((2*u + 2*j*d)/(6*j*d) + v^2/(36*j^2*d^2))^3)^(1/2) - v/(4*j*d) - v^3/(216*j^3*d^3) - (v*(2*u + 2*j*d))/(24*j^2*d^2))^(1/3)) + (3^(1/2)*((((v/(4*j*d) + v^3/(216*j^3*d^3) + (v*(2*u + 2*j*d))/(24*j^2*d^2))^2 - ((2*u + 2*j*d)/(6*j*d) + v^2/(36*j^2*d^2))^3)^(1/2) - v/(4*j*d) - v^3/(216*j^3*d^3) - (v*(2*u + 2*j*d))/(24*j^2*d^2))^(1/3) - ((2*u + 2*j*d)/(6*j*d) + v^2/(36*j^2*d^2))/(((v/(4*j*d) + v^3/(216*j^3*d^3) + (v*(2*u + 2*j*d))/(24*j^2*d^2))^2 - ((2*u + 2*j*d)/(6*j*d) + v^2/(36*j^2*d^2))^3)^(1/2) - v/(4*j*d) - v^3/(216*j^3*d^3) - (v*(2*u + 2*j*d))/(24*j^2*d^2))^(1/3))*i)/2 - v/(6*j*d);
