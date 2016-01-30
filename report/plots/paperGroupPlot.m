function paperGroupPlot(X1, YMatrix1, YMatrix2, YMatrix3)
%CREATEFIGURE(X1, YMATRIX1, YMATRIX2, YMATRIX3)
%  X1:  vector of x data
%  YMATRIX1:  matrix of y data
%  YMATRIX2:  matrix of y data
%  YMATRIX3:  matrix of y data

%  Auto-generated by MATLAB on 18-Jan-2016 23:32:23

% Create figure
figure1 = figure;

% Create axes
axes1 = axes('Parent',figure1,'XTickLabel',{'0','10','30','50','70','90'},...
    'XTick',[1 2 3 4 5 6],...
    'FontWeight','bold',...
    'FontSize',20,...
    'Position',[0.13 0.299979123173271 0.226951154052603 0.405657620041747]);
%% Uncomment the following line to preserve the X-limits of the axes
xlim(axes1,[0 7]);
%% Uncomment the following line to preserve the Y-limits of the axes
% ylim(axes1,[0 0.25]);
box(axes1,'on');
hold(axes1,'on');

% Create multiple lines using matrix input to plot
plot1 = plot(X1,YMatrix1,'Parent',axes1,'Marker','diamond','LineWidth',2);
set(plot1(1),'DisplayName','ConcatNMF',...
    'Color',[0.635294117647059 0.0784313725490196 0.184313725490196]);
set(plot1(2),'DisplayName','WorstView',...
    'Color',[0.301960784313725 0.745098039215686 0.933333333333333]);
set(plot1(3),'DisplayName','BestView',...
    'Color',[0.466666666666667 0.674509803921569 0.188235294117647]);
set(plot1(4),'DisplayName','PairwiseSC');
set(plot1(5),'DisplayName','CentroidSC','Color',[1 0.8 0.2]);
set(plot1(6),'DisplayName','PVC',...
    'Color',[0.850980392156863 0.325490196078431 0.0980392156862745]);
set(plot1(7),'DisplayName','GPMVC',...
    'Color',[0 0.447058823529412 0.741176470588235]);

% Create xlabel
xlabel('PER (%)','FontSize',20);

% Create ylabel
ylabel('NMI','FontSize',20);

% Create title
title('Cora');

% Create axes
axes2 = axes('Parent',figure1,'YTick',[0.2 0.4 0.6 0.8 1],...
    'XTickLabel',{'0','10','30','50','70','90'},...
    'XTick',[1 2 3 4 5 6],...
    'FontWeight','bold',...
    'FontSize',20,...
    'Position',[0.410797101449275 0.298914842985819 0.222589908749329 0.405980261909286]);
%% Uncomment the following line to preserve the X-limits of the axes
xlim(axes2,[0 7]);
box(axes2,'on');
hold(axes2,'on');

% Create multiple lines using matrix input to plot
plot2 = plot(X1,YMatrix2,'Parent',axes2,'Marker','diamond','LineWidth',2);
set(plot2(1),'DisplayName','GPMVC');
set(plot2(2),'DisplayName','PVC','Color',[0.85 0.325 0.098]);
set(plot2(3),'DisplayName','CentroidSC','Color',[0.929 0.694 0.125]);
set(plot2(4),'DisplayName','PairwiseSC');
set(plot2(5),'DisplayName','BestView','Color',[0.466 0.674 0.188]);
set(plot2(6),'DisplayName','WorstView');
set(plot2(7),'DisplayName','ConcatNMF');

% Create xlabel
xlabel('PER (%)');

% Create ylabel
ylabel('NMI');

% Create title
title('Digit');

% Create legend
legend1 = legend(axes2,'show');
set(legend1,...
    'Position',[0.130434782608696 0.783978325339582 0.782608695652174 0.0712041884816754],...
    'Orientation','horizontal',...
    'FontSize',23);

% Create axes
axes3 = axes('Parent',figure1,'YTick',[0.4 0.6 0.8 1],...
    'XTickLabel',{'0','10','30','50','70','90'},...
    'XTick',[1 2 3 4 5 6],...
    'FontWeight','bold',...
    'FontSize',20,...
    'Position',[0.687600644122383 0.299958684321933 0.225442834138486 0.401482254697285]);
%% Uncomment the following line to preserve the X-limits of the axes
xlim(axes3,[0 7]);
%% Uncomment the following line to preserve the Y-limits of the axes
% ylim(axes3,[0.2 1]);
box(axes3,'on');
hold(axes3,'on');

% Create multiple lines using matrix input to plot
plot3 = plot(X1,YMatrix3,'Parent',axes3,'Marker','diamond','LineWidth',2);
set(plot3(1),'DisplayName','GPMVC');
set(plot3(2),'DisplayName','PVC',...
    'Color',[0.929411764705882 0.694117647058824 0.125490196078431]);
set(plot3(3),'DisplayName','CentroidSC',...
    'Color',[0.494117647058824 0.184313725490196 0.556862745098039]);
set(plot3(4),'DisplayName','PairwiseSC',...
    'Color',[0.850980392156863 0.325490196078431 0.0980392156862745]);
set(plot3(5),'DisplayName','BestView','Color',[0.466 0.674 0.188]);
set(plot3(6),'DisplayName','WorstView');
set(plot3(7),'DisplayName','ConcatNMF','Color',[0.635 0.078 0.184]);

% Create xlabel
xlabel('PER (%)');

% Create ylabel
ylabel('NMI');

% Create title
title('ORL');
