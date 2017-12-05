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

figure;
hold on;
p1 = scatter(wine1(:,1), wine1(:,2), markerSize, 'r', 'filled');
p2 = scatter(wine2(:,1), wine2(:,2), markerSize, 'b', 'filled');
p3 = scatter(wine3(:,1), wine3(:,2), markerSize, 'g', 'filled');
alpha(p1, .5);
alpha(p2, .5);
alpha(p3, .5);
hold off;
title("LDA (Wine Dataset)");

end