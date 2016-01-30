function finalPaperPlots(X1, YMatrix1)
%CREATEFIGURE(X1, YMATRIX1)
%  X1:  vector of x data
%  YMATRIX1:  matrix of y data

%  Auto-generated by MATLAB on 17-Jan-2016 18:16:23

% Create figure
figure1 = figure;

% Create axes
axes1 = axes('Parent',figure1,'XTickLabel',{'0','10','30','50','70','90'},...
    'XTick',[1 2 3 4 5 6],...
    'FontWeight','bold',...
    'FontSize',18,...
    'Position',[0.241909554482019 0.146988795518207 0.533923778851315 0.781932773109244]);
%% Uncomment the following line to preserve the X-limits of the axes
xlim(axes1,[0 7]);
%% Uncomment the following line to preserve the Y-limits of the axes
ylim(axes1,[0 1]);
box(axes1,'on');
hold(axes1,'on');

% Create multiple lines using matrix input to plot
plot1 = plot(X1,YMatrix1,'MarkerSize',8,'Marker','diamond','LineWidth',2,...
    'Parent',axes1);
set(plot1(1),'DisplayName','GPMVC');
set(plot1(2),'DisplayName','CentroidSC');
set(plot1(3),'DisplayName','PairwiseSC');
set(plot1(4),'DisplayName','MultiNMF');
set(plot1(5),'DisplayName','BestView');
set(plot1(6),'DisplayName','WorstView');
set(plot1(7),'DisplayName','ConcatNMF');

% Create xlabel
xlabel('PER (%)','FontSize',15.4);

% Create ylabel
ylabel('NMI','FontSize',15.4);

% Create title
title('ORL');

% Create legend
legend1 = legend(axes1,'show');
set(legend1,...
    'Position',[0.246947129026567 0.854636291248864 0.526033269738358 0.0471204188481675],...
    'Orientation','horizontal',...
    'FontSize',14);
