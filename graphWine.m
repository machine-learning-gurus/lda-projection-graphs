function graphWine = graphWine()

data = dlmread("wine.csv",",");
X = data(:,2:end); y = data(:,1);

% Principal Component
[Dpca, Wpca] = pca(X);
Xm = bsxfun(@minus, X, mean(X));
Xproj = project(Xm, Wpca(:,1:2));

% Linear Discriminant Analysis
[Dlda,Wlda] = lda(X,y);
Xproj = project(Xm, Wlda(:,1:2));

wine1 = Xproj(find(y == 1), :);
wine2 = Xproj(find(y == 2), :);
wine3 = Xproj(find(y == 3), :);
markerSize = 20;

wine1_avg = mean([wine1(:,1), wine1(:,2)]);
wine2_avg = mean([wine2(:,1), wine2(:,2)]);
wine3_avg = mean([wine3(:,1), wine3(:,2)]);

set(gcf,'color','w');
figure;
hold on;
p1 = scatter(wine1(:,1), wine1(:,2), markerSize, 'r', 'filled');
p1_avg = scatter(wine1_avg(:,1), wine1_avg(:,2), 15000, 'r', 'filled');
p2 = scatter(wine2(:,1), wine2(:,2), markerSize, 'y', 'filled');
p2_avg = scatter(wine2_avg(:,1), wine2_avg(:,2), 20000, 'y', 'filled');
p3 = scatter(wine3(:,1), wine3(:,2), markerSize, 'g', 'filled');
p3_avg = scatter(wine3_avg(:,1), wine3_avg(:,2), 15000, 'g', 'filled');
alpha(p1, .5);
alpha(p1_avg, .25);
alpha(p2, .5);
alpha(p2_avg, .25);
alpha(p3, .5);
alpha(p3_avg, .25);
hold off;

set(gca, 'Color', [50 56 62] ./ 255);
title("LDA Projection (Wine Dataset)");

end