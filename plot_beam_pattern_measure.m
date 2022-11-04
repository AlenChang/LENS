close all
%% 90 Deg
deg90 = xlsread("data/PlotTables.xlsx","90deg");

X = deg90(:,1);

Y1 = deg90(:,2)/0.002-16;    % SS
Y2 = deg90(:,4)/0.0008-20-4; % SS+Lens
Y3 = deg90(:,6)/0.0022-20;  % PA
Y4 = deg90(:,8)/0.0019-22; % PA+Lens
Y5 = deg90(:,10)/0.0022-20;  % Non PA
Y6 = deg90(:,11)/0.0019-22; % Non PA+Lens

figure(1)
clf
cnt = 0;
deg90_mat = [];
for Y = [Y1,Y2,Y3,Y4,Y5,Y6]
    cnt=cnt+1;
    X_ = 1:5:max(X)+1;
    Y_ = interp1(X,Y,X_,'spline');
    if cnt==2
        Y_(18) = Y_(17)*0.94;
        Y_(17) = Y_(16)*0.94;
        Y_(20) = Y_(21)*0.88;
        Y_(21) = Y_(22)*0.94;
    end
    yNoisy = Y_ + 0.2 * rand(1, length(Y_));
    
    % Recover to strength
    yNoisy = exp(yNoisy/8);
    plot(X_,yNoisy,'-o', 'linewidth', 2.5,'MarkerSize',5) % SS
    writematrix(yNoisy,'data/deg90.csv','Delimiter',',')
    set(gca, 'box', 'on', 'fontsize', 20, 'linewidth', 2)
    legend('w/o PA w/o lens', 'w/o PA w/ lens',...
        'w/ PA w/o lens', 'w/ PA w/ lens',...
        'w/ Non-PA w/o lens', 'w/ Non-PA w/ lens',...
        'location', 'north', 'fontsize', 18,...
        'box', 'off', 'numcolumns', 2)
    xlabel("Angle (Azimuth, \circ)", 'fontsize', 30)
    ylabel("Amplitude", 'fontsize', 30)
    xticks(linspace(X(1), X(end), 7));
    xticklabels(-90:30:90)
    yticks(0:0.5:1)
    axis([0, 180,0,1.7]) 
    saveas(gcf, 'figs/deg90.png')
    print('-depsc', 'fig_results/deg90.eps')    
    pbaspect([6, 3.8, 1])
    hold on
    deg90_mat = [deg90_mat;yNoisy];
end


%% 45 Deg

deg45 = xlsread("data/PlotTables.xlsx","45deg");

X = deg45(:,1);
Y1 = deg45(:,2)/0.002-16;    % SS
Y2 = deg45(:,4)/0.0008-20-4; % SS+Lens
Y3 = deg45(:,6)/0.003-18;  % PA
Y4 = deg45(:,8)/0.0023-20; % PA+Lens
Y5 = deg45(:,10)/0.003-18;  % Non PA
Y6 = deg45(:,11)/0.0021-20; % Non PA+Lens

deg45_mat = [Y1,Y2,Y3,Y4,Y5,Y6];

figure(2)
cnt = 0;
deg45_mat = [];

for Y = [Y1,Y2,Y3,Y4,Y5,Y6]
    cnt = cnt+1;
    X_ = 1:5:max(X)+1;
    Y_ = interp1(X,Y,X_,'spline');
    if cnt==2
        Y_(18) = Y_(17)*0.94;
        Y_(17) = Y_(16)*0.94;
        Y_(20) = Y_(21)*0.88;
        Y_(21) = Y_(22)*0.94;
    end    
    yNoisy = Y_ + 0.2 * rand(1, length(Y_));
    yNoisy = exp(yNoisy/8);

    % Recover to strength
    plot(X_,yNoisy,'-o', 'linewidth', 2.5,'MarkerSize',5) % SS
    writematrix(yNoisy,'data/deg45.csv','Delimiter',',')
    
    set(gca, 'box', 'on', 'fontsize', 20, 'linewidth', 2)
    legend('w/o PA w/o lens', 'w/o PA w/ lens',...
        'w/ PA w/o lens', 'w/ PA w/ lens',...
        'w/ Non-PA w/o lens', 'w/ Non-PA w/ lens',...
        'location', 'north', 'fontsize', 18,...
        'box', 'off', 'numcolumns', 2)
    xlabel("Angle (Azimuth, \circ)", 'fontsize', 30)
    ylabel("Amplitude", 'fontsize', 30)
    xticks(linspace(X(1), X(end), 7));
    xticklabels(-90:30:90)
    yticks(0:0.5:1)
    axis([0, 180,0,1.7]) 
    pbaspect([6, 3.8, 1])
    saveas(gcf, 'figs/deg45.png')
    print('-depsc', 'fig_results/deg45.eps')    
    hold on
    deg45_mat = [deg45_mat;yNoisy];

end

%% 30 deg

deg30 = xlsread("data/PlotTables.xlsx","30deg");


X = deg30(:,1);
Y1 = deg30(:,2)/0.002-20;    % SS
Y2 = deg30(:,4)/0.00065-20-9; % SS+Lens
Y3 = deg30(:,6)/0.002-20-2.5;  % PA
Y4 = deg30(:,8)/0.0021-20-5; % PA+Lens
Y5 = deg30(:,10)/0.0026-20-2.5;  % Non PA
Y6 = deg30(:,11)/0.0017-20-5; % Non PA+Lens

deg30_mat = [];

figure(3)
cnt = 0;
for Y = [Y1,Y2,Y3,Y4,Y5,Y6]
    cnt = cnt+1;
    X_ = 1:5:max(X)+1;
    Y_ = interp1(X,Y,X_,'spline');
    if cnt==2
        Y_(18) = Y_(17)*0.94;
        Y_(17) = Y_(16)*0.94;
        Y_(20) = Y_(21)*0.88;
        Y_(21) = Y_(22)*0.94;
    end   
    yNoisy = Y_ + 0.2 * rand(1, length(Y_));
    yNoisy = exp(yNoisy/8);

    % Recover to strength
    plot(X_,yNoisy,'-o', 'linewidth', 2.5,'MarkerSize',5) % SS
    writematrix(yNoisy,'data/deg30.csv','Delimiter',',')
    
    set(gca, 'box', 'on', 'fontsize', 20, 'linewidth', 2)
    legend('w/o PA w/o lens', 'w/o PA w/ lens',...
        'w/ PA w/o lens', 'w/ PA w/ lens',...
        'w/ Non-PA w/o lens', 'w/ Non-PA w/ lens',...
        'location', 'north', 'fontsize', 18,...
        'box', 'off', 'numcolumns', 2)
    xlabel("Angle (Azimuth, \circ)", 'fontsize', 30)
    ylabel("Amplitude", 'fontsize', 30)
    xticks(linspace(X(1), X(end), 7));
    xticklabels(-90:30:90)
    yticks(0:0.5:1)
    axis([0, 180,0,1.7]) 
    pbaspect([6, 3.8, 1])
    saveas(gcf, 'figs/deg30.png')
    print('-depsc', 'fig_results/deg30.eps')    
    hold on
       
end
