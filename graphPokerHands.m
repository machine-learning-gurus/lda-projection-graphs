function graphPokerHands = graphPokerHands()

% Load Wine Dataset
data = dlmread("poker-hand-training-true.data",",");
y = data(:,end);
X = data(:,1:end-1);

% Principal Component
[Dpca, Wpca] = pca(X);
Xm = bsxfun(@minus, X, mean(X));

% Linear Discriminant Analysis
[Dlda,Wlda] = lda(X,y);
Xproj = project(Xm, Wlda(:,1:2));

c0 = Xproj(find(y == 0), :);
c1 = Xproj(find(y == 1), :);
c2 = Xproj(find(y == 2), :);
c3 = Xproj(find(y == 3), :);
c4 = Xproj(find(y == 4), :);
c5 = Xproj(find(y == 5), :);
c6 = Xproj(find(y == 6), :);
c7 = Xproj(find(y == 7), :);
c8 = Xproj(find(y == 8), :);
c9 = Xproj(find(y == 9), :);

figure;
hold on;

c0_size = size(c0);
markerSize = 20;

p1 = scatter(c0(:,1), c0(:,2), markerSize, 'r', 'filled');
p2 = scatter(c1(:,1), c1(:,2), markerSize, 'b', 'filled');
p3 = scatter(c3(:,1), c3(:,2), markerSize, 'g', 'filled');
p4 = scatter(c4(:,1), c4(:,2), markerSize, 'm', 'filled');
p5 = scatter(c5(:,1), c5(:,2), markerSize, 'c', 'filled');
p6 = scatter(c6(:,1), c6(:,2), markerSize, 'k', 'filled');
p7 = scatter(c7(:,1), c7(:,2), markerSize, [.2 .2 .2], 'filled');
p8 = scatter(c8(:,1), c8(:,2), markerSize, [1 .5 .2], 'filled');
p9 = scatter(c9(:,1), c9(:,2), markerSize, [.2 .5 1], 'filled');
alpha(p1, .25);
alpha(p2, .25);
alpha(p3, .25);
alpha(p4, .25);
alpha(p5, .25);
alpha(p6, .25);
alpha(p7, .25);
alpha(p8, .25);
alpha(p9, .25);
hold off;
title("LDA (Poker Hand Dataset)");

end