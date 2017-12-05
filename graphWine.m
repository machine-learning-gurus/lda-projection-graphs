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

figure;
hold on;
plot(wine1(:,1), wine1(:,2), "ro", "markersize", 3, "linewidth", 3);
plot(wine2(:,1), wine2(:,2), "go", "markersize", 3, "linewidth", 3);
plot(wine3(:,1), wine3(:,2), "bo", "markersize", 3, "linewidth", 3);
title("LDA (original data)")

end