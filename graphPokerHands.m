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

c0_avg = mean([c0(:,1), c0(:,2)]);
c1_avg = mean([c1(:,1), c1(:,2)]);
c2_avg = mean([c2(:,1), c2(:,2)]);
c3_avg = mean([c3(:,1), c3(:,2)]);
c4_avg = mean([c4(:,1), c4(:,2)]);
c5_avg = mean([c5(:,1), c5(:,2)]);
c6_avg = mean([c6(:,1), c6(:,2)]);
c7_avg = mean([c7(:,1), c7(:,2)]);
c8_avg = mean([c8(:,1), c8(:,2)]);
c9_avg = mean([c9(:,1), c9(:,2)]);

p1 = scatter(c0(:,1), c0(:,2), markerSize, 'r', 'filled');
p1_avg = scatter(c0_avg(:,1), c0_avg(:,2), 55000, 'r', 'filled');
p2 = scatter(c1(:,1), c1(:,2), markerSize, 'b', 'filled');
p2_avg = scatter(c1_avg(:,1), c1_avg(:,2), 65000, 'b', 'filled');
p3 = scatter(c3(:,1), c3(:,2), markerSize, 'g', 'filled');
p3_avg = scatter(c3_avg(:,1), c3_avg(:,2), 45000, 'g', 'filled');
p4 = scatter(c4(:,1), c4(:,2), markerSize, 'm', 'filled');
p4_avg = scatter(c4_avg(:,1), c4_avg(:,2), 75000, 'm', 'filled');
p5 = scatter(c5(:,1), c5(:,2), markerSize, 'c', 'filled');
p5_avg = scatter(c5_avg(:,1), c5_avg(:,2), 50000, 'c', 'filled');
p6 = scatter(c6(:,1), c6(:,2), markerSize, 'k', 'filled');
p6_avg = scatter(c6_avg(:,1), c6_avg(:,2), 40000, 'k', 'filled');
p7 = scatter(c7(:,1), c7(:,2), markerSize, [.2 .2 .2], 'filled');
p7_avg = scatter(c7_avg(:,1), c7_avg(:,2), 10000, [.2 .2 .2], 'filled');
p8 = scatter(c8(:,1), c8(:,2), markerSize, [1 .5 .2], 'filled');
p8_avg = scatter(c8_avg(:,1), c8_avg(:,2), 5000, [1 .5 .2], 'filled');
p9 = scatter(c9(:,1), c9(:,2), markerSize, [.2 .5 1], 'filled');
p9_avg = scatter(c9_avg(:,1), c9_avg(:,2), 3000, [.2 .5 1], 'filled');
alpha(p1, .15);
alpha(p1_avg, .3);
alpha(p2, .15);
alpha(p2_avg, .3);
alpha(p3, .15);
alpha(p3_avg, .3);
alpha(p4, .15);
alpha(p4_avg, .3);
alpha(p5, .25);
alpha(p5_avg, .3);
alpha(p6, .15);
alpha(p6_avg, .3);
alpha(p7, .15);
alpha(p7_avg, .3);
alpha(p8, .15);
alpha(p8_avg, .3);
alpha(p9, .15);
alpha(p9_avg, .3);
hold off;

set(gca, 'Color', [50 56 62] ./ 255);
title("LDA Projection (Poker Hand Dataset)");

end