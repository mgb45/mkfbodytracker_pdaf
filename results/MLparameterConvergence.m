close all
clear all
clc;

load('GammaL3D.txt')
load('GammaR3D.txt')
load('GammaL3D_PCA.txt')
load('GammaR3D_PCA.txt')

figure;
subplot (2,2,1)
plot(1:1000:50000,GammaL3D(:,1:50)','LineWidth',2)
xlabel('Samples')
ylabel('\gamma')
title('Left Arm')
axis([-inf inf,0.5 1])
grid on
subplot (2,2,2)
plot(1:1000:50000,GammaR3D(:,1:50)','LineWidth',2)
xlabel('Samples')
ylabel('\gamma')
title('Right Arm')
axis([-inf inf,0.5 1])
grid on
subplot (2,2,3)
plot(1:1000:50000,GammaL3D_PCA','LineWidth',2)
xlabel('Samples')
ylabel('\gamma')
title('Left Arm, PCA')
axis([-inf inf,0.5 1])
grid on
subplot (2,2,4)
plot(1:1000:50000,GammaR3D_PCA','LineWidth',2)
xlabel('Samples')
ylabel('\gamma')
title('Right Arm, PCA')
axis([-inf inf,0.5 1])
grid on

