clear; clc; close all
% Transmitted Power
TxPower_dBmW = 24;% dBm;
%% Active Band parameters
% For practical AE {0.75-1, 1.25-1.4, 1.7-2.1, 3.2-4.4} THz
% StartFreqTHz = [0.75,1.25,1.7,3.2];
% StopFreqTHz = [1,1.4,2.1,4.4];
StartFreqTHz = 0.75 ; % 0.25272;
StopFreqTHz = 4.4 ; % 0.32184;
%% Trajectory parameters
RealSimulatedFlag = 1;% 0: Real, 1: Simulated
LinearSphereFlag = 0;% 0: Linear, 1: Sphere
ThetaPhiFlag = 0;% 0: Theta Direction, 1: Phi Direction
StraightDiagonalFlag = 0;% 0: Straight, 1: Diagonal
%% Setting inital corrdinates for drones
x_t = 0; y_t = 10; z_t = 100;
x_r = 0; y_r = 11; z_r = 100; % for theta/phi, d = 20 m
% x_r = 0; y_r = 10; z_r = 120; % for theta/phi, d = 20 m
% x_r = 0; y_r = 11; z_r = 100; For linear/diagonal
%% Antenna parameters
% Receiver
AEtypeRx = 3;
% BeamWidthRx_deg = 10;
% SideLobeStrengthRx = 0.05;
BeamWidthRx_deg = 10;
SideLobeStrengthRx = 0.1;
% Transmitter
AEtypeTx = AEtypeRx;
% BeamWidthTx_deg = 10;
% SideLobeStrengthTx = 0.05;
BeamWidthTx_deg = 10;%0.181;
SideLobeStrengthTx = 0.1;

% Antenna positions on drone
% Transmitter
thetaTxBoreSight_deg = -90;
phiTxBoreSight_deg = 90;
% Receiver
thetaRxBoreSight_deg = 90;
phiRxBoreSight_deg = 90;
%% Data of accelerations Tx Rx
if RealSimulatedFlag == 0
    load data_small_windy_average.mat
% vi = 0; % for small scale mobility case, i.e., just hovering
dt = mean(diff(DataNewAv(:,1)));% %5 msec sampling period
timeFinal = 5.5;
FinalIdx = round(timeFinal/dt);
LastIdx = floor(length(DataNewAv(:,1))/FinalIdx)*FinalIdx;
Data = DataNewAv(1:LastIdx,:);
DataR = reshape(Data,[FinalIdx,LastIdx/FinalIdx,size(Data,2)]);
Trace1Number = 2;
Trace2Number = 1;
DataTx = squeeze(DataR(:,Trace1Number,:));
DataRx = squeeze(DataR(:,Trace2Number,:));

acc_x_t = DataTx(:,2);
acc_y_t = DataTx(:,3);
acc_z_t = DataTx(:,4);
time_mob_t = DataTx(:,1)-DataTx(1,1);

idx_r = FinalIdx+1:FinalIdx*2;

acc_x_r = DataRx(:,2);
acc_y_r = DataRx(:,3);
acc_z_r = DataRx(:,4);
time_mob_r = DataRx(:,1)-DataRx(1,1);

%     load(uigetfile);
elseif RealSimulatedFlag == 1
    DataTx = [(0:10)',zeros(11,3)];
    DataRx = [(0:10)',zeros(11,3)];
else
    error('RealSimulatedFlag == 0,1')
end

%% Using linear accelereations
if LinearSphereFlag == 0 
time_mob_t  = DataTx(:,1);
acc_x_t     = DataTx(:,2);
acc_y_t     = DataTx(:,3);
acc_z_t     = DataTx(:,4);
vi_x_t      = 0;
vi_y_t      = 0;
vi_z_t      = 0;

time_mob_r  = DataRx(:,1);
acc_x_r     = DataRx(:,2);
acc_y_r     = DataRx(:,3);
acc_z_r     = DataRx(:,4);

if StraightDiagonalFlag == 0
    vi_x_r      = 0;
    vi_y_r      = 10;
elseif StraightDiagonalFlag == 1
    vi_x_r      = sqrt(2)*5;
    vi_y_r      = sqrt(2)*5;
else
    error('StraightDiagonalFlag == 0,1')
end
vi_z_r      = 0;
% Using Accelerations for constant trajectory
[~,x_mobility_t] = a2vd(acc_x_t,time_mob_t,vi_x_t,x_t);
[~,y_mobility_t] = a2vd(acc_y_t,time_mob_t,vi_y_t,y_t);
[~,z_mobility_t] = a2vd(acc_z_t,time_mob_t,vi_z_t,z_t);

[~,x_mobility_r] = a2vd(acc_x_r,time_mob_r,vi_x_r,x_r);
[~,y_mobility_r] = a2vd(acc_y_r,time_mob_r,vi_y_r,y_r);
[~,z_mobility_r] = a2vd(acc_z_r,time_mob_r,vi_z_r,z_r);

%% Using sphere motion
elseif LinearSphereFlag == 1 
    if RealSimulatedFlag == 1
        % x_t = 0; y_t = 10; z_t = 100;
        % x_r = 0; y_r = 11; z_r = 100;
        if ThetaPhiFlag == 0
            theta_d = 0:10:180;%[0:10:180];
            phi_d = [90];
        elseif ThetaPhiFlag == 1
            theta_d = 180;%180;%[0:10:180];
            phi_d = 0:10:180;%[90];
        else
            error('ThetaPhiFlag == 0,1');
        end
        r = sqrt(sum((x_t-x_r)^2+(y_t-y_r)^2+(z_t-z_r)^2));%meters
        XYZ_Tx = [x_t,y_t,z_t];
        XYZ_Rx = [x_r,y_r,z_r];
        
        XYZ = GenHemSphCoord(r,theta_d,phi_d,XYZ_Tx,XYZ_Rx);
        x_mobility_t = XYZ(:,1);
        y_mobility_t =  XYZ(:,2);
        z_mobility_t =  XYZ(:,3);
        x_mobility_r =  XYZ(:,4);
        y_mobility_r =  XYZ(:,5);
        z_mobility_r =  XYZ(:,6);
        time_mob_t = (0:(length(z_mobility_r)-1))*1;
    else
        error('RealSimulatedFlag == 1')
    end
else
    error('LinearSphereFlag == 0,1')
end
%% Plotting of figures
% figure(1);
% plot3(x_mobility_t,y_mobility_t,z_mobility_t,'*','MarkerSize',10)
% hold on
% plot3(x_mobility_r,y_mobility_r,z_mobility_r,'.','MarkerSize',20)
% xlabel('x coordinate [m]')
% ylabel('y coordinate [m]')
% zlabel('z coordinate [m]')
% title('Tx Fixed - Rx Moving away with Constant Velocity')
% set(gca,'FontSize',14)
% legend('Tx Drone Trail','Rx Drone Trail','FontSize',12)
% grid on;axis square;
% drawnow
% hold off
%% 
d_t_r_mob = sqrt((x_mobility_r-x_mobility_t).^2+...
    (y_mobility_r-y_mobility_t).^2+...
    (z_mobility_r-z_mobility_t).^2);

% figure(2);
% plot(time_mob_t,d_t_r_mob,'-*','LineWidth',5,'MarkerSize',10)
% xlabel('Time (t) [sec]')
% ylabel({'Instantaneous distance (d_{mob})';'between Tx & Rx drones [m]'})
% title('Tx Fixed - Rx Moving away with Constant Velocity')
% set(gca,'FontSize',14)
% grid on;axis square;
% legend('Tx-Rx','FontSize',12)
% drawnow
%%
DataOutT = zeros(length(x_mobility_t),1);

% Calcualte intial antenna alignment
cTx = [x_mobility_t,y_mobility_t,z_mobility_t];
cRx = [x_mobility_r,y_mobility_r,z_mobility_r];
[thetaTxBoreSight_deg,phiTxBoreSight_deg, ...
    thetaRxBoreSight_deg,phiRxBoreSight_deg] = BorsightCalc(cTx(1,:),cRx(1,:));
% Drone coordinates
FreqTHz_TxT = cell(length(x_mobility_t),1);
% figure(3)
for kk = 1:length(x_mobility_t)
    CoordTx = [x_mobility_t(kk),y_mobility_t(kk),z_mobility_t(kk)];
    CoordRx = [x_mobility_r(kk),y_mobility_r(kk),z_mobility_r(kk)];
    BandWidthOptFlag = 0;% 0: STD, 1: OA, 2: CFB
    BeamWidthOptFlag = 0;% 0: WBO, 1: BO
    PowerAllocFlag = 1;% 0: EP, 1: WF

    %% 
    
alphaMAx = 1;%150;
for alpha = 1:alphaMAx
   [CapOutxx,FreqTHz_Tx,BeamWidthRx_deg_max,...
    BeamWidthTx_deg_max,SNRout_dB,GainTotal] = Distance_to_Cap_Opt(...
        PowerAllocFlag,TxPower_dBmW,...
        CoordRx,CoordTx,...
        StartFreqTHz,StopFreqTHz,...
        AEtypeRx,BeamWidthRx_deg,SideLobeStrengthRx,...
        AEtypeTx,BeamWidthTx_deg,SideLobeStrengthTx,...
        thetaTxBoreSight_deg,phiTxBoreSight_deg,...
        thetaRxBoreSight_deg,phiRxBoreSight_deg,...
        BandWidthOptFlag,BeamWidthOptFlag);
    CapacityT(alpha) = CapOutxx;
end
CapOut = mean(CapacityT);

    %%
%     [CapOut,FreqTHz_Tx,BeamWidthRx_deg_max,...
%     BeamWidthTx_deg_max,SNRout_dB,GainTotal] = Distance_to_Cap_Opt(...
%         PowerAllocFlag,TxPower_dBmW,...
%         CoordRx,CoordTx,...
%         StartFreqTHz,StopFreqTHz,...
%         AEtypeRx,BeamWidthRx_deg,SideLobeStrengthRx,...
%         AEtypeTx,BeamWidthTx_deg,SideLobeStrengthTx,...
%         thetaTxBoreSight_deg,phiTxBoreSight_deg,...
%         thetaRxBoreSight_deg,phiRxBoreSight_deg,...
%         BandWidthOptFlag,BeamWidthOptFlag);
    %% 
    % Rx Delta Theta and Phi
    [thetaRx_deg,phiRx_deg] = Coord2ThetaPhi(CoordTx,CoordRx);
    delta_thetaRx_deg = AcuteAngle(thetaRxBoreSight_deg,thetaRx_deg);
%     delta_thetaRx_deg = mod(abs(thetaRxBoreSight_deg - thetaRx_deg),180);
%     delta_phiRx_deg = abs(phiRxBoreSight_deg - phiRx_deg);
    delta_phiRx_deg = AcuteAngle(phiRxBoreSight_deg,phiRx_deg);
    % Tx Delta Theta and Phi
    [thetaTx_deg,phiTx_deg] = Coord2ThetaPhi(CoordRx,CoordTx);
%     delta_thetaTx_deg = mod(abs(thetaTxBoreSight_deg - thetaTx_deg),180);
    delta_thetaTx_deg = AcuteAngle(thetaTxBoreSight_deg,thetaTx_deg);
%     delta_phiTx_deg = abs(phiTxBoreSight_deg - phiTx_deg);
    delta_phiTx_deg = AcuteAngle(phiTxBoreSight_deg,phiTx_deg);
    %% 
    FreqTHz_TxT{kk,1} = FreqTHz_Tx;
    DataOutT(kk,1:8) = [CapOut,BeamWidthRx_deg_max,BeamWidthTx_deg_max,...
        GainTotal,delta_thetaRx_deg,delta_phiRx_deg,delta_thetaTx_deg,delta_phiTx_deg];
%     plot(time_mob_t(1:kk),DataOutT(1:kk,1)/1e9);drawnow;
end

% Plot Freq vs time
% figure(4)
% hold off
% MrkrSize= 10;
% for ll = 1:length(FreqTHz_TxT)
%     freqTHz_ll = FreqTHz_TxT{ll};
%     time_ll = ones(size(freqTHz_ll))*time_mob_t(ll);
%     plot(time_ll,freqTHz_ll,'.','MarkerSize',MrkrSize+2)
% hold on
% end
% xlabel('Time (t) [sec]')
% ylabel('Freq (THz)')
% title('Active Frequency Bands wrt Distance')
% set(gca,'FontSize',14)
% grid on;axis square;
% hold off
% 
% % Plot Average Capacity
% figure(5);
% semilogy(time_mob_t,DataOutT(:,1)/1e9,'g','LineWidth',5)
% xlabel('Time (t) [sec]')
% ylabel('Capacity (C) [Gbps]')
% title('Tx Fixed - Rx Moving away with Constant Velocity')
% set(gca,'FontSize',14)
% legend({'Tx (Fixed) - Rx Moving Away'},'FontSize',12)
% grid on;axis square;
% drawnow

%% Beam Width VS Time
figure(6);
yyaxis right
hold off
yyaxis left
hold off
plot(time_mob_t,DataOutT(:,2),'.-y','LineWidth',9)
hold on
plot(time_mob_t,DataOutT(:,2),':k','LineWidth',6)
hold on
plot(time_mob_t,DataOutT(:,3),'-b','LineWidth',3)
hold on
plot(time_mob_t,DataOutT(:,3),'--r','LineWidth',3)
hold on
plot(time_mob_t,DataOutT(:,6),'--co','LineWidth',3,'MarkerSize',6)
hold on
plot(time_mob_t,DataOutT(:,5),'--gd','LineWidth',3,'MarkerSize',6)
hold on
ylabel('Angles [degrees]')
yyaxis right
plot(time_mob_t,DataOutT(:,4),'--r*','LineWidth',3,'MarkerSize',12)
set(gca,'FontSize',14)
xlabel('Time (t) [sec]')
ylabel('Total Antenna Gain (G_{Total}) [dB]')
title('Tx Fixed - Rx Moving away with Constant Velocity')
legend({'\psi_{\phi - Tx}','\psi_{\theta - Tx}',...
    '\psi_{\phi - Rx}','\psi_{\theta - Rx}',...
    '\Delta{\phi}_{\{Rx,Tx\}}','\Delta{\theta}_{\{Rx,Tx\}}',...
    'G_{Total \{Rx,Tx\}}'},'FontSize',10)
grid on;axis square;
hold off;
drawnow
%% Vs Distance
% Plot Freq vs time
% figure(7)
% MrkrSize= 10;
% for ll = 1:length(FreqTHz_TxT)
%     freqTHz_ll = FreqTHz_TxT{ll};
%     d_t_r_mob_ll = ones(size(freqTHz_ll))*d_t_r_mob(ll);
%     plot(d_t_r_mob_ll,freqTHz_ll,'.','MarkerSize',MrkrSize+2)
% hold on
% end
% xlabel('Distance [meters]')
% ylabel('Freq (THz)')
% title('Active Frequency Bands wrt Distance')
% set(gca,'FontSize',14)
% grid on;axis square;
% hold off

%%
% Plot Capacity vs Distance
figure(8);
semilogy(d_t_r_mob,DataOutT(:,1)/1e9,'-r+','LineWidth',1)
xlabel('Distance [meters]')
ylabel('Capacity (C) [Gbps]')
title('Tx Fixed - Rx Moving away with Constant Velocity')
set(gca,'FontSize',14)
% legend({'Tx (Fixed) - Rx Moving Away'},'FontSize',12)
grid on;axis square;
hold on
drawnow

%% Beam Width VS Distance
% figure(9);
% plot(d_t_r_mob,DataOutT(:,2:3),'LineWidth',7)
% hold on
% plot(d_t_r_mob,DataOutT(:,2:3),'LineWidth',3)
% set(gca,'FontSize',14)
% xlabel('Distance [meters]')
% ylabel('Beamwidth (deg)')
% title('Tx Fixed - Rx Moving away with Constant Velocity')
% legend({'Beamwidth - Rx','Beamwidth - Tx'},'FontSize',12)
% grid on;axis square;
% drawnow
% hold off;
% %% Spectral Efficiency
% NB_count = cellfun(@length,FreqTHz_TxT);
% Total_Band = (NB_count.*3.120000008493652e+08);
% SE = DataOutT(:,1)./Total_Band;
% %%
% % Plot SE vs Distance
% figure(10);
% semilogy(d_t_r_mob,SE,'-y','LineWidth',1)
% xlabel('Distance [meters]')
% ylabel('Spectral Efficiency (SE) [bits/sec/Hz]')
% title('Tx Fixed - Rx Moving away with Constant Velocity')
% set(gca,'FontSize',14)
% % legend({'Tx (Fixed) - Rx Moving Away'},'FontSize',12)
% grid on;axis square;
% hold on
% drawnow
% %%
% % Plot SE vs Time
% figure(11);
% plot(time_mob_t,SE,'-y','LineWidth',1)
% xlabel('Time [sec]')
% ylabel('Spectral Efficiency (SE) [bits/sec/Hz]')
% title('Tx Fixed - Rx Moving')
% set(gca,'FontSize',14)
% % legend({'Tx (Fixed) - Rx Moving Away'},'FontSize',12)
% grid on;axis square;
% hold on
% drawnow
% %% Beam Width VS Distance
% figure(12);
% plot(time_mob_t,DataOutT(:,2:3),'LineWidth',7)
% hold on
% plot(time_mob_t,DataOutT(:,2:3),'LineWidth',3)
% set(gca,'FontSize',14)
% xlabel('Time [sec]')
% ylabel('Beamwidth (deg)')
% title('Tx Fixed - Rx Moving')
% legend({'Beamwidth - Rx','Beamwidth - Tx'},'FontSize',12)
% grid on;axis square;
% drawnow
% hold off;

% Capacity vs. Theta
% figure(13);
% semilogy(XYZ(:,7),DataOutT(:,1)/1e9,'g','LineWidth',5)
% xlabel('Azimuth Angle (\theta) [degrees]')
% ylabel('Capacity (C) [Gbps]')
% title('Tx Fixed - Rx Moving away')
% set(gca,'FontSize',14)
% legend({'Tx (Fixed) - Rx Moving Away'},'FontSize',12)
% grid on;axis square;
% drawnow
% 
% Capacity vs. Elevation
% x
% 
% figure(14);
% semilogy(XYZ(:,8),SE,'-y','LineWidth',1)
% xlabel('Elevation Angle (\phi) [degrees]')
% ylabel('Spectral Efficiency (SE) [bits/sec/Hz]')
% title('Tx Fixed - Rx Moving away with Constant Velocity')
% set(gca,'FontSize',14)
% legend({'Tx (Fixed) - Rx Moving Away'},'FontSize',12)
% grid on;axis square;
% hold on
% drawnow

%% Energy Efficiency
% figure(15);
% semilogy(time_mob_t,((DataOutT(:,1))/1e9)/0.25,'-r','LineWidth',1)
% xlabel('Time (t) [sec]')
% ylabel('Energy Efficiency (EE) [Giga bits/joules]')
% title('Tx - Rx Moving for With Practical Traces')
% set(gca,'FontSize',14)
% % legend({'Tx (Fixed) - Rx Moving Away'},'FontSize',12)
% grid on;axis square;
% hold on
% drawnow
% figure(15);
% semilogy(time_mob_t,((DataOutT(:,1))/1e9)/0.25,'-r','LineWidth',1)
% xlabel('Time (t) [sec]')
% ylabel('Energy Efficiency (EE) [Giga bits/joules]')
% title('Tx - Rx Moving for With Practical Traces')
% set(gca,'FontSize',14)
% % legend({'Tx (Fixed) - Rx Moving Away'},'FontSize',12)
% grid on;axis square;
% hold on
% drawnow
% d_t_r_mob
% figure(16);
% semilogy(XYZ(:,8),((DataOutT(:,1))/1e9)/0.25,'-r','LineWidth',1)
% % xlabel('Zenith Angle (\phi) [degrees]')
% xlabel('Elevation Angle (\phi) [degrees]')
% ylabel('Energy Efficiency (EE) [Giga bits/joules]')
% title('Tx - Rx Moving for With Practical Traces')
% set(gca,'FontSize',14)
% % legend({'Tx (Fixed) - Rx Moving Away'},'FontSize',12)
% grid on;axis square;
% hold on
drawnow

%% Helping Functions
%% 
function [tTx,pTx, tRx,pRx] = BorsightCalc(cTx,cRx)
[tRx,pRx,~] = Coord2ThetaPhi(cTx,cRx);
% dVect_12 = cTx-cRx;
% d_12 = sqrt(sum(dVect_12.^2));
% pRx = acosd(dVect_12(3)./d_12);
% tRx = atan2d(dVect_12(2),dVect_12(1));

[tTx,pTx,~] = Coord2ThetaPhi(cRx,cTx);
% dVect_21 = cRx-cTx;
% d_21 = sqrt(sum(dVect_21.^2));
% pTx = acosd(dVect_21(3)./d_21);
% tTx = atan2d(dVect_21(2),dVect_21(1));
end
%%
function [v,d] = a2vd(a,t,v0,d0)
% Find velocity and distance at a given linear accelerations for each time
% stamp
N = length(a);
if length(t)~=N
    error('acc  ~=  time')
end
v = zeros(size(a));
d = zeros(size(a));
v(1) = v0;
d(1) = d0;
for kk = 2:N
    a_kk = (a(2:kk)+a(1:kk-1))/2;
    dt_kk = (t(2:kk)-t(1:kk-1));
    v(kk) = sum(a_kk.*dt_kk)+v0;
    d(kk) = d(kk-1) + v(kk-1)*dt_kk(end) + 0.5*a_kk(end)*dt_kk(end).^2;
end
end
%%
function [Theta21_deg,Phi21_deg,d] = Coord2ThetaPhi(Coord1,Coord2)
dVect_12 = Coord1-Coord2;
d = sqrt(sum(dVect_12.^2));
Phi21_deg = acosd(dVect_12(3)./d);
Theta21_deg = mod(360+atan2d(dVect_12(2),dVect_12(1)),360);
% Theta21_deg = atan2d(dVect_12(2),dVect_12(1)); Original by Ammar Bhai
end
%% 
function [Freq_THz,AbsorptionLoss_dB,PLtotal_dB,Gain_dB,TxPower_dBW] = ...
                        Distance_to_Loss(TxPower_dBmW,...
                        CoordRx,CoordTx,...
                        AEtypeRx,BeamWidthRx_deg,SideLobeStrengthRx,...
                        AEtypeTx,BeamWidthTx_deg,SideLobeStrengthTx,...
                        thetaTxBoreSight_deg,phiTxBoreSight_deg,...
                        thetaRxBoreSight_deg,phiRxBoreSight_deg)
%% Derived variables
TxPower_dBW = TxPower_dBmW-30;

[thetaRx_deg,phiRx_deg,~] = Coord2ThetaPhi(CoordTx,CoordRx);
[thetaTx_deg,phiTx_deg,d] = Coord2ThetaPhi(CoordRx,CoordTx);
%% Calcualte antenna Gain
% Transmitter Gain 
GainTx = AEmodel(thetaTx_deg, phiTx_deg,thetaTxBoreSight_deg, phiTxBoreSight_deg,...
    BeamWidthTx_deg,SideLobeStrengthTx,AEtypeTx);
GainTx_dB = pow2db(GainTx);
% Receiver Gain
GainRx = AEmodel(thetaRx_deg, phiRx_deg,thetaRxBoreSight_deg, phiRxBoreSight_deg,...
    BeamWidthRx_deg,SideLobeStrengthRx,AEtypeRx);
GainRx_dB = pow2db(GainRx);
% Total Gain
Gain_dB = GainTx_dB + GainRx_dB;

%% Channel Effects
H1 = CoordRx(3)/1000;H2 = CoordTx(3)/1000;
ZANGLE = phiRx_deg;RANGE_km = d/1000;
Season = 6;
DisplayFlag = 0;VerboseFlag = 0;
[Freq_THz,~,~,AbsorptionLoss_dB,PLtotal_dB,~] = ...
    LBLRTM_FullOutput(H1,H2,ZANGLE,RANGE_km,Season,...
    DisplayFlag,VerboseFlag);
end
%% 
function [Cap,ReqFreq_THz,BeamWidthRx_deg_max,...
    BeamWidthTx_deg_max,SNRout_dB,Gain_dB] = Distance_to_Cap_Opt(...
                        PowerAllocFlag,TxPower_dBmW,...
                        CoordRx,CoordTx,...
                        StartFreqTHz,StopFreqTHz,...
                        AEtypeRx,BeamWidthRx_deg,SideLobeStrengthRx,...
                        AEtypeTx,BeamWidthTx_deg,SideLobeStrengthTx,...
                        thetaTxBoreSight_deg,phiTxBoreSight_deg,...
                        thetaRxBoreSight_deg,phiRxBoreSight_deg,...
                        BandWidthOptFlag,BeamWidthOptFlag)
%% Error calculate
if length(StartFreqTHz)~=length(StopFreqTHz)
    error('length(StartFreqTHz)==length(StopFreqTHz). should satisfy')
end
%% Distance_to_Loss
[Freq_THz,~,~,~,~] = ...
                        Distance_to_Loss(TxPower_dBmW,...
                        CoordRx,CoordTx,...
                        AEtypeRx,BeamWidthRx_deg,SideLobeStrengthRx,...
                        AEtypeTx,BeamWidthTx_deg,SideLobeStrengthTx,...
                        thetaTxBoreSight_deg,phiTxBoreSight_deg,...
                        thetaRxBoreSight_deg,phiRxBoreSight_deg);
%% Active frequency region
ReqIdx = [];
for ii = 1:length(StartFreqTHz)
  Idx_ii = find((Freq_THz>=StartFreqTHz(ii))&...
                        (Freq_THz<=StopFreqTHz(ii)));
  ReqIdx = [ReqIdx;Idx_ii];
end
ReqIdx_Init_Orig = unique(ReqIdx);

maskIdx_Init_Orig = zeros(size(Freq_THz));
maskIdx_Init_Orig(ReqIdx_Init_Orig) = 1;
%% Gain optimization
if BeamWidthOptFlag == 0
    BeamWidthRx_deg_max = BeamWidthRx_deg;
    BeamWidthTx_deg_max = BeamWidthTx_deg;
    SideLobeStrengthRx_max = SideLobeStrengthRx;
    SideLobeStrengthTx_max = SideLobeStrengthTx;
elseif BeamWidthOptFlag == 1
    [BeamWidthRx_deg_max,BeamWidthTx_deg_max,...
        SideLobeStrengthRx_max,SideLobeStrengthTx_max] = ...
        AntennaGainOptim(BeamWidthRx_deg,BeamWidthTx_deg,...
        SideLobeStrengthRx,SideLobeStrengthTx,TxPower_dBmW,...
        CoordRx,CoordTx, AEtypeRx,AEtypeTx,...
        thetaTxBoreSight_deg,phiTxBoreSight_deg,...
        thetaRxBoreSight_deg,phiRxBoreSight_deg,...
        maskIdx_Init_Orig);
else
    error('BeamWidthOptFlag == 0,1')
end
[Freq_THz,AbsorptionLoss_dB_GainOpt,...
    PLtotal_dB_GainOpt,Gain_dB_GainOpt,TxPower_dBW_GainOpt] = ...
            Distance_to_Loss(TxPower_dBmW,...
            CoordRx,CoordTx,...
            AEtypeRx,BeamWidthRx_deg_max,SideLobeStrengthRx_max,...
            AEtypeTx,BeamWidthTx_deg_max,SideLobeStrengthTx_max,...
            thetaTxBoreSight_deg,phiTxBoreSight_deg,...
            thetaRxBoreSight_deg,phiRxBoreSight_deg);
%%
maskIdx_Init = maskIdx_Init_Orig;
maskIdx = maskIdx_Init;
maskOut = maskIdx_Init;
distance_m = sqrt(sum((CoordRx-CoordTx).^2));
AbsorptionLoss_dB = AbsorptionLoss_dB_GainOpt;
PLtotal_dB = PLtotal_dB_GainOpt;
Gain_dB = Gain_dB_GainOpt;
TxPower_dBW = TxPower_dBW_GainOpt;
while 1
    %% Power Allocaiton
    maskOutOld = maskOut;
    if PowerAllocFlag == 0
        PowerNBProdNsub_dBW = pow2db(maskIdx_Init.*db2pow(TxPower_dBW+Gain_dB));
        maskOut = maskIdx_Init;
    elseif PowerAllocFlag == 1
        % Water Filling approach
        [PowerNBProdNsub_dBW,maskOut] = WaterFillingAlgo(maskIdx_Init,...
            AbsorptionLoss_dB,PLtotal_dB,...
            TxPower_dBW+Gain_dB,Freq_THz);
    else
        error('PowerAllocFlag == 0,1');
    end
    %% BandWidth Optimization
    ReqIdx_Init = find(maskOut==1);   
    maskIdxOld = maskIdx;
    if BandWidthOptFlag == 0
        maskIdx = maskOut;
    elseif BandWidthOptFlag == 1
        maskIdx = ActiveFreqMask(maskOut,Freq_THz,...
            PLtotal_dB,AbsorptionLoss_dB,0,PowerNBProdNsub_dBW,ReqIdx_Init);
    elseif BandWidthOptFlag == 2
        maskIdx = CommonFlatBandFreqMask(maskOut,Freq_THz,...
            PLtotal_dB,AbsorptionLoss_dB,0,PowerNBProdNsub_dBW,ReqIdx_Init,...
            distance_m);
    else
        error('BandWidthOptFlag == 0,1')
    end  
    % Stopiing criteria
    A = find(maskIdx);
    B = find(maskOut);
    if isempty(setdiff(union(A,B),intersect(A,B)))
        FlagBreak = 1;
        break;
    end
%     A = find(maskIdx);
%     B = find(maskIdxOld);
%     if isempty(setdiff(union(A,B),intersect(A,B)))
%         FlagBreak = 2;
%         break;
%     end
%     A = find(maskOut);
%     B = find(maskOutOld);
%     if isempty(setdiff(union(A,B),intersect(A,B)))
%         FlagBreak = 3;
%         break;
%     end
    maskIdx_Init = maskIdx;
end
if FlagBreak == 1
    [Cap,ReqFreq_THz,SNRout_dB] = Mask_Cap(maskOut,Freq_THz,...
        PLtotal_dB,AbsorptionLoss_dB,0,PowerNBProdNsub_dBW);
elseif FlagBreak == 2
    PowerNBProdNsub_W = (db2pow(PowerNBProdNsub_dBW).*maskIdx);
    PowerNBProdNsub_W = db2pow(TxPower_dBW+Gain_dB)/sum(PowerNBProdNsub_W).*PowerNBProdNsub_W;
    PowerNBProdNsub_dBW = pow2db(PowerNBProdNsub_W);
    [Cap,ReqFreq_THz,SNRout_dB] = Mask_Cap(maskIdx,Freq_THz,...
        PLtotal_dB,AbsorptionLoss_dB,0,PowerNBProdNsub_dBW);
elseif FlagBreak == 3
    [Cap,ReqFreq_THz,SNRout_dB] = Mask_Cap(maskOut,Freq_THz,...
        PLtotal_dB,AbsorptionLoss_dB,0,PowerNBProdNsub_dBW);
else
    error('Invalid FlagBreak, 0 1 2')
end
end
%% Antenna Gain Optimization
function [BeamWidthRx_deg_max,BeamWidthTx_deg_max,...
    SideLobeStrengthRx_max,SideLobeStrengthTx_max] = ...
    AntennaGainOptim(BeamWidthRx_deg,BeamWidthTx_deg,...
    SideLobeStrengthRx,SideLobeStrengthTx,TxPower_dBmW,...
            CoordRx,CoordTx, AEtypeRx,AEtypeTx,...
            thetaTxBoreSight_deg,phiTxBoreSight_deg,...
            thetaRxBoreSight_deg,phiRxBoreSight_deg,...
            maskIdx)
% Low res
BeamWidthRx_deg_range = linspace(BeamWidthRx_deg,360,15);
BeamWidthTx_deg_range = linspace(BeamWidthTx_deg,360,15);
xData_Cap = ...
    zeros(length(BeamWidthRx_deg_range)*length(BeamWidthTx_deg_range),5);
kk = 1;
for BeamWidthRx_deg_kk = BeamWidthRx_deg_range
    for BeamWidthTx_deg_kk = BeamWidthTx_deg_range
        if BeamWidthRx_deg_kk == BeamWidthTx_deg_kk
            SideLobeStrengthRx_kk = SideLobeStrengthRx;
            SideLobeStrengthTx_kk = SideLobeStrengthTx;

            [Freq_THz_kk,AbsorptionLoss_dB_kk,PLtotal_dB_kk,...
                Gain_dB_kk,TxPower_dBW_kk] = ...
                Distance_to_Loss(TxPower_dBmW,...
                CoordRx,CoordTx,...
                AEtypeRx,BeamWidthRx_deg_kk,SideLobeStrengthRx_kk,...
                AEtypeTx,BeamWidthTx_deg_kk,SideLobeStrengthTx_kk,...
                thetaTxBoreSight_deg,phiTxBoreSight_deg,...
                thetaRxBoreSight_deg,phiRxBoreSight_deg);
            [Cap_kk,~,~] = Mask_Cap(maskIdx,Freq_THz_kk,...
                PLtotal_dB_kk,AbsorptionLoss_dB_kk,Gain_dB_kk,TxPower_dBW_kk);
            xData_Cap(kk,:) = [BeamWidthRx_deg_kk,BeamWidthTx_deg_kk,...
                SideLobeStrengthRx_kk,SideLobeStrengthTx_kk,Cap_kk];
            kk = kk + 1;
        end
    end
end
xData_Cap = xData_Cap(1:kk-1,:);
[~,Idx_max_cap] = max(xData_Cap(:,5),[],1);
BeamWidthRx_deg_crude = xData_Cap(Idx_max_cap,1);
BeamWidthTx_deg_crude = xData_Cap(Idx_max_cap,2);
SideLobeStrengthRx_crude = xData_Cap(Idx_max_cap,3);
SideLobeStrengthTx_crude = xData_Cap(Idx_max_cap,4);








% Hi res
BmWdStepRx = mean(diff(BeamWidthRx_deg_range));
BeamWidthRx_deg_range = BeamWidthRx_deg_crude + ...
    (-BmWdStepRx-1:2:BmWdStepRx+1);
BeamWidthRx_deg_range(BeamWidthRx_deg_range<BeamWidthRx_deg) = [];
BeamWidthRx_deg_range(BeamWidthRx_deg_range>360) = [];

BmWdStepTx = mean(diff(BeamWidthTx_deg_range));
BeamWidthTx_deg_range = BeamWidthTx_deg_crude + ...
    (-BmWdStepTx-1:2:BmWdStepTx+1);
BeamWidthTx_deg_range(BeamWidthTx_deg_range<BeamWidthTx_deg) = [];
BeamWidthTx_deg_range(BeamWidthTx_deg_range>360) = [];

xData_Cap = ...
    zeros(length(BeamWidthRx_deg_range)*length(BeamWidthTx_deg_range),5);

kk = 1;
for BeamWidthRx_deg_kk = BeamWidthRx_deg_range
    for BeamWidthTx_deg_kk = BeamWidthTx_deg_range
        if BeamWidthRx_deg_kk == BeamWidthTx_deg_kk
            SideLobeStrengthRx_kk = SideLobeStrengthRx_crude;
            SideLobeStrengthTx_kk = SideLobeStrengthTx_crude;
            
            [Freq_THz_kk,AbsorptionLoss_dB_kk,PLtotal_dB_kk,Gain_dB_kk,TxPower_dBW_kk] = ...
                Distance_to_Loss(TxPower_dBmW,...
                CoordRx,CoordTx,...
                AEtypeRx,BeamWidthRx_deg_kk,SideLobeStrengthRx_kk,...
                AEtypeTx,BeamWidthTx_deg_kk,SideLobeStrengthTx_kk,...
                thetaTxBoreSight_deg,phiTxBoreSight_deg,...
                thetaRxBoreSight_deg,phiRxBoreSight_deg);
            [Cap_kk,ReqFreq_THz_kk,SNRout_dB_kk] = Mask_Cap(maskIdx,Freq_THz_kk,...
                PLtotal_dB_kk,AbsorptionLoss_dB_kk,Gain_dB_kk,TxPower_dBW_kk);
            xData_Cap(kk,:) = [BeamWidthRx_deg_kk,BeamWidthTx_deg_kk,...
                SideLobeStrengthRx_kk,SideLobeStrengthTx_kk,Cap_kk];
            kk = kk +1;
        end
    end
end
xData_Cap = xData_Cap(1:kk-1,:);

[~,Idx_max_cap] = max(xData_Cap(:,5),[],1);
BeamWidthRx_deg_max = xData_Cap(Idx_max_cap,1);
BeamWidthTx_deg_max = xData_Cap(Idx_max_cap,2);
SideLobeStrengthRx_max = xData_Cap(Idx_max_cap,3);
SideLobeStrengthTx_max = xData_Cap(Idx_max_cap,4);
end
%% Optimize Frequency Band for Max Capacity
function maskIdx = ActiveFreqMask(maskIdx_Init,Freq_THz,...
    PLtotal_dB,AbsorptionLoss_dB,Gain_dB,TxPower_dBW,ReqIdx_Init)
[Cap_Init,ReqFreq_THz_Init,SNRout_Init] = Mask_Cap(maskIdx_Init,Freq_THz,...
    PLtotal_dB,AbsorptionLoss_dB,Gain_dB,TxPower_dBW);
ReqIdx_Init = find(maskIdx_Init==1);
Cap_Old = 0;
SNRout_Sort = sort(SNRout_Init,'descend');
StartIdx = 1;
xData = [];yData = [];
TxPower_W = db2pow(TxPower_dBW)/length(ReqIdx_Init);
TotalTxPower_W = sum(TxPower_W);
% crude
for jj = StartIdx:100:length(SNRout_Sort)
    maskIdx = zeros(size(Freq_THz));
    SNR_jj = SNRout_Sort(jj);
    Idx_jj = find(SNRout_Init >=SNR_jj);
    ReqIdx_jj = ReqIdx_Init(Idx_jj);
    maskIdx(ReqIdx_jj) = 1;
    TxPower_W_jj = zeros(size(maskIdx));
    TxPower_W_jj(ReqIdx_jj) = TxPower_W(ReqIdx_jj)/sum(TxPower_W(ReqIdx_jj))*TotalTxPower_W*length(ReqIdx_jj);
    TxPower_dBW_jj = pow2db(TxPower_W_jj);
    
    [Cap_jj,ReqFreq_THz_jj,SNRout_jj] = Mask_Cap(maskIdx,Freq_THz,...
    PLtotal_dB,AbsorptionLoss_dB,Gain_dB,TxPower_dBW_jj);
    xData = [xData,jj];
    yData = [yData,Cap_jj];
%     plot(xData,yData,'.');drawnow;
end
% to cater for the last value that might not be included in the crude
jj = length(SNRout_Sort);
maskIdx = zeros(size(Freq_THz));
SNR_jj = SNRout_Sort(jj);
Idx_jj = find(SNRout_Init >=SNR_jj);
ReqIdx_jj = ReqIdx_Init(Idx_jj);
maskIdx(ReqIdx_jj) = 1;

TxPower_W_jj = zeros(size(maskIdx));
    TxPower_W_jj(ReqIdx_jj) = TxPower_W(ReqIdx_jj)/sum(TxPower_W(ReqIdx_jj))*TotalTxPower_W*length(ReqIdx_jj);
    TxPower_dBW_jj = pow2db(TxPower_W_jj);
    
%
[Cap_jj,ReqFreq_THz_jj,SNRout_jj] = Mask_Cap(maskIdx,Freq_THz,...
    PLtotal_dB,AbsorptionLoss_dB,Gain_dB,TxPower_dBW_jj);
xData = [xData,jj];
yData = [yData,Cap_jj];
% plot(xData,yData,'.');drawnow;

%
[~,maxIdx_ydata] = max(yData);
maxIdx = xData(maxIdx_ydata);
StartIdx = maxIdx-100-StartIdx+1;
if StartIdx<1
    StartIdx = 1;
end
xData = [];yData = [];
if maxIdx+100 > length(SNRout_Sort)
    EndIdx = length(SNRout_Sort);
else
    EndIdx = maxIdx+100;
end
for jj = StartIdx:1:EndIdx
    maskIdx = zeros(size(Freq_THz));
    SNR_jj = SNRout_Sort(jj);
    Idx_jj = find(SNRout_Init >=SNR_jj);
    ReqIdx_jj = ReqIdx_Init(Idx_jj);
    maskIdx(ReqIdx_jj) = 1;
    TxPower_W_jj = zeros(size(maskIdx));
    TxPower_W_jj(ReqIdx_jj) = TxPower_W(ReqIdx_jj)/sum(TxPower_W(ReqIdx_jj))*TotalTxPower_W*length(ReqIdx_jj);
    TxPower_dBW_jj = pow2db(TxPower_W_jj);
    
    % 
    [Cap_jj,ReqFreq_THz_jj,SNRout_jj] = Mask_Cap(maskIdx,Freq_THz,...
    PLtotal_dB,AbsorptionLoss_dB,Gain_dB,TxPower_dBW_jj);
    xData = [xData,jj];
    yData = [yData,Cap_jj];
%     plot(xData,yData,'.');drawnow;
end
maskIdx = zeros(size(Freq_THz));
[~,maxIdx_yData] = max(yData);
Idx_Max = xData(maxIdx_yData);
SNR_max = SNRout_Sort(Idx_Max);
Idx_MaxT = find(SNRout_Init >=SNR_max);
ReqIdx_Max = ReqIdx_Init(Idx_MaxT);
maskIdx(ReqIdx_Max) = 1;
    % 
%     [Cap,ReqFreq_THz,SNRout_dB] = Mask_Cap(maskIdx,Freq_THz,...
%     PLtotal_dB,AbsorptionLoss_dB,Gain_dB,TxPower_dBW);
% plot(ReqFreq_THz,SNRout_dB,'.');drawnow;
% hold on
% plot(ReqFreq_THz_Init,pow2db(SNRout_Init),'.');drawnow;
end
%% Common Flat Band Frequency Band for Max Capacity
function maskIdx = CommonFlatBandFreqMask(maskIdx_Init,Freq_THz,...
    PLtotal_dB,AbsorptionLoss_dB,Gain_dB,TxPower_dBW,ReqIdx_Init,dist_m)

% % [Idx_Total_PLpNoise_Flat] = FlatBandCapacityCalculator(dist_m,...
% %                             AbsorptionLoss_dB,Freq_THz,ReqIdx_Init);
% [Idx_Total_PLpNoise_Flat] = FlatBandCapacityCalculator(dist_m,...
%                             AbsorptionLoss_dB,Freq_THz,ReqIdx_Init,PLtotal_dB);
% 
% maskPL_NoiseFlat = zeros(size(maskIdx_Init),'like',maskIdx_Init);
% maskPL_NoiseFlat((Idx_Total_PLpNoise_Flat)) = 1;
% maskIdx = maskIdx_Init & maskPL_NoiseFlat;
%% 
if dist_m ==1
    load d1m2.mat
elseif dist_m ==11
load d11m.mat 
elseif dist_m ==21
load d21m3.mat
elseif dist_m ==31
load d31m2.mat
elseif dist_m ==41
load d41m2.mat
elseif dist_m ==51
load d51m2.mat
elseif dist_m ==61
load d61m2.mat
elseif dist_m ==71
load d71m3.mat
elseif dist_m ==81
load d81m2.mat
elseif dist_m ==91
load d91m2.mat
elseif dist_m ==101
load d101m2.mat
end
%%
end
%% Calculate masking to capacity with active frquencies
function [Cap,ReqFreq_THz,SNRout] = Mask_Cap(maskIdx,Freq_THz,...
    PLtotal_dB,AbsorptionLoss_dB,Gain_dB,TxPower_dBW)
Idx = find(maskIdx==1);
DeltaFreq_Hz = DeltaFreq(Freq_THz*1e12);
ReqFreq_THz = Freq_THz(Idx);
ReqDeltaFreq = DeltaFreq_Hz(Idx);

if length(TxPower_dBW) == 1
    ReqTxPower_dBW = TxPower_dBW;
else
    ReqTxPower_dBW = TxPower_dBW(Idx);
end
ReqPLtotal_dB = PLtotal_dB(Idx);
ReqAbsorptionLoss_dB = AbsorptionLoss_dB(Idx);
SNRout = PL_Noise2SNR(ReqPLtotal_dB,ReqAbsorptionLoss_dB,...
                        Gain_dB,ReqTxPower_dBW,ReqDeltaFreq);
Cap = sum(ReqDeltaFreq(:).*log2(1+SNRout(:)));
end
%% Actute Angle calculator
function AngleOut = AcuteAngle(AngleIn1,AngleIn2)
diffAngle = mod(abs(AngleIn1-AngleIn2),360);
if diffAngle > 180 && diffAngle <= 360 
    AngleOut = 360 - diffAngle;
elseif diffAngle <= 180 && diffAngle >= 0 
    AngleOut = diffAngle;
else
    error('Wrong angles, there is some problem somwhere ...')
end
end
%% Antenna Model
function Gain = AEmodel(thetaIn, phiIn,thetaBoreSight, phiBoreSight,...
    BeamWidth,SideLobeStrength,AEtype)
% AEtype : 0 - Isotropic
%          1 - Omin-directional
%          2 - Sector AE 3-D assumed
%          3 - 
    if length(BeamWidth) == 1
        BeamWidth = [BeamWidth,BeamWidth];
    elseif length(BeamWidth) > 2 || length(BeamWidth) < 1
        error('Wrong paraemter for length(BeamWidth)')
    end
%     delta_theta = mod(abs(360+thetaBoreSight - thetaIn),360);
if phiBoreSight == 0 || phiBoreSight == 180
    delta_theta = 0;
else
    delta_theta = AcuteAngle(thetaBoreSight,thetaIn);    
end
%     delta_theta = mod(abs(thetaBoreSight - thetaIn),180); %Original
delta_phi = AcuteAngle(phiBoreSight,phiIn);
%     delta_phi = abs(phiBoreSight - phiIn);
    Cond = @(t,p) double((BeamWidth(1)/2 >= t)&&(BeamWidth(2)/2 >=p));
%     Cond = @(t,p) double((BeamWidth(1)/2 >= t)&&(BeamWidth(1)/2 >=p));
    switch AEtype
        case 0 % Isotropic antenna
            Gain = 1;
        case 1 % Omni directional antenna
            Gain = 2;
        case 2 % WCNC conference Antenna Model
            BmW = sum(deg2rad(BeamWidth));
            Gain = SideLobeStrength*(1-Cond(delta_theta,delta_phi)) + ...
                    ((2*pi - (2*pi - BmW)*SideLobeStrength) ./ BmW) .* ...
                    Cond(delta_theta,delta_phi);
        case 3 % Spherical Lobe Model
            BmW_Phi = (deg2rad(BeamWidth(1)));
            BmW_theta = (deg2rad(BeamWidth(2)));
            Gain = SideLobeStrength*(1-Cond(delta_theta,delta_phi)) + ...
                    4*pi/(BmW_Phi*BmW_theta).* ...
                    Cond(delta_theta,delta_phi);
        case 4 % Spherical Lobe Model with side lobe gain subtraction
            BmW_Phi = (deg2rad(BeamWidth(1)));
            BmW_theta = (deg2rad(BeamWidth(2)));
            Gain = SideLobeStrength*(1-Cond(delta_theta,delta_phi)) + ...
                    (4*pi-(4*pi-BmW_Phi-BmW_theta)*SideLobeStrength)/(BmW_Phi*BmW_theta).* ...
                    Cond(delta_theta,delta_phi);
        case 5 % path (microstrip) antenna
            Gain = 6; %6 to 9 dBi
        otherwise
            Gain = 1;
    end
end
%% Delta Freq
function deltaFreq = DeltaFreq(Freq)
if length(Freq)==1
    deltaFreq = 3.120000008493652e+08;
else
    deltaFreq = diff(Freq);
    deltaFreq = [deltaFreq(1);deltaFreq]/2+[deltaFreq;deltaFreq(end)]/2;
end
end
%% PL+Noise to SNR
function [SNRout] = PL_Noise2SNR(Atotal_dB,AbsLoss_dB,Gain_dB,P_tx_dB,deltaFreq)
% Path Loss Calculations
Atotal_dB_Dummy = Atotal_dB;
Atotal_dB_Dummy(Atotal_dB_Dummy==Inf)=[];
Atotal_dB_DummyMax = max(Atotal_dB_Dummy);
Atotal_dB(find(Atotal_dB>=Atotal_dB_DummyMax)) = Atotal_dB_DummyMax;
Atotal = db2pow(Atotal_dB);
% Noise power calculations
Pnoise = NoisePower(AbsLoss_dB,deltaFreq);
% Equal Power Allocation
Gain = db2pow(Gain_dB);
P_tx = db2pow(P_tx_dB);
P_Total = P_tx(:).*Gain(:);
N_subBands = length(deltaFreq);
TransmitPower = P_Total/N_subBands;
% SNR calculations
SNRout = TransmitPower(:)./(Atotal(:).*Pnoise(:));
end
%% Noise Power calcualtions
function Pnoise = NoisePower(AbsLoss_dB,deltaFreq)
if length(AbsLoss_dB)~=length(deltaFreq)
    error('Check the dimentions of AbsLoss_dB, deltaFreq')
end
Trans = 1./db2pow(AbsLoss_dB);
Trans(find(Trans>=1)) = 0.9999999999;
Epsilon = 1 - Trans;
T0 = 296;
Tmol = T0 * Epsilon;
Tnoise = Tmol;
kB = 1.3806e-23;
Pnoise = kB.*Tnoise.*deltaFreq;
end

%% Calculate Path Loss
function [Freq_THz,Trans,AbsCoeff,AbsorptionLoss_dB,PLtotal_dB,SR] = ...
    LBLRTM_FullOutput(H1,H2,ZANGLE,RANGE_km,Season,DisplayFlag,VerboseFlag)
%% Load Database Files here using if elseif
% load LBLRTM_H1_0.1_H2_0.1_ZANGLE_90_RANGE_km_0.011_Season_6_data.mat
if RANGE_km == 1e-3
load LBLRTM_H1_0.1_H2_0.1_ZANGLE_90_RANGE_km_0.001_Season_6_data.mat
elseif RANGE_km == 0.011
load LBLRTM_H1_0.1_H2_0.1_ZANGLE_90_RANGE_km_0.011_Season_6_data.mat
elseif RANGE_km == 0.021
load LBLRTM_H1_0.1_H2_0.1_ZANGLE_90_RANGE_km_0.021_Season_6_data.mat
elseif RANGE_km == 0.031
load LBLRTM_H1_0.1_H2_0.1_ZANGLE_90_RANGE_km_0.031_Season_6_data.mat
elseif RANGE_km == 0.041
load LBLRTM_H1_0.1_H2_0.1_ZANGLE_90_RANGE_km_0.041_Season_6_data.mat
elseif RANGE_km == 0.051
load LBLRTM_H1_0.1_H2_0.1_ZANGLE_90_RANGE_km_0.051_Season_6_data.mat
elseif RANGE_km == 0.061
load LBLRTM_H1_0.1_H2_0.1_ZANGLE_90_RANGE_km_0.061_Season_6_data.mat
elseif RANGE_km == 0.071
load LBLRTM_H1_0.1_H2_0.1_ZANGLE_90_RANGE_km_0.071_Season_6_data.mat
elseif RANGE_km == 0.081
load LBLRTM_H1_0.1_H2_0.1_ZANGLE_90_RANGE_km_0.081_Season_6_data.mat
elseif RANGE_km == 0.091
load LBLRTM_H1_0.1_H2_0.1_ZANGLE_90_RANGE_km_0.091_Season_6_data.mat
elseif RANGE_km == 0.101
load LBLRTM_H1_0.1_H2_0.1_ZANGLE_90_RANGE_km_0.101_Season_6_data.mat
end

% dataOutput = LBLRTM_Transmittance(H1,H2,ZANGLE,RANGE_km,Season,...
%     DisplayFlag,VerboseFlag);
% Transmittance
lambda_cm_inv = dataOutput.v;
lambda_cm = 1./(lambda_cm_inv);
lambda_meters = lambda_cm.*1e-2;
Freq_Hz = 3e8./lambda_meters;
Freq_THz = Freq_Hz./1e12;
Trans = dataOutput.tau; %tau transmittance
% Trans(Trans<0)=0;
if ~isempty(Trans(Trans<0))
    error('Transmittance is less than zero at this settings')
end
SR = dataOutput.SlantRangef_km;
SR = SR.*1000;
% Absorption Coefficients K(f)
minTransVal = min(Trans(Trans~=0))*1e-10;
Trans(Trans==0) = minTransVal;
AbsCoeff = (-log(Trans))./SR;
% Total Absorption Loss [dB]
AbsorptionLoss_dB = ...
    AbsCoeff.*SR.*10.*log10(2.71828);
% Total Path Loss [dB]
PLspread_dB = 20.*log10(SR) + ...
    20.*log10(Freq_Hz./1e6)-27.55;%-GtdB-GrdB;
PLabs_dB = AbsorptionLoss_dB;
PLtotal_dB = PLspread_dB + PLabs_dB;

end
%% LBLRTM call and transmittance
function dataOut = LBLRTM_Transmittance(H1,H2,ZANGLE,RANGE_km,Season,DisplayFlag,VerboseFlag)
% Directories
CodePath = cd;%'F:\Capacity_Analysis\LBLRTM_Test';
addpath(CodePath);% 'F:\Capacity_Analysis\LBLRTM_Test'
DirLbl = ['C:\MinGW\msys\1.0\home\AKHTARSU\',...
    'akhtar\lnfl_lblrtm_aer_new\lblrtm\']; %C:\lblrtm\
% Command for running LBLRTM
runlblrtm = ['C:\MinGW\msys\1.0\home\AKHTARSU\akhtar\',...
    'lnfl_lblrtm_aer_new\lblrtm\lblrtm_v12.2_mingw_gnu_sgl'] ;

% G:\LBLRTM Son\Zip Check\lblrtm
% DirLbl = ['G:\LBLRTM Son\Zip Check\lblrtm\'];
% % Command for running LBLRTM
% runlblrtm = ['G:\LBLRTM Son\Zip Check\lblrtm\lblrtm_v12.2_mingw_gnu_sgl'] ;


cd(DirLbl)
% Main Code Starts from here
if ZANGLE > 90
    % We are looking up
    if H2 > H1
        cd(CodePath)
        error('H1 should be greater than H2 if ZANGLE is greater than 90')        
    end
else
    if H2 < H1
        cd(CodePath)
        error('H2 should be greater than H1 if ZANGLE is less than 90')        
    end
end
% Defining the type of transmission horizontal or slant
Cond = abs(H1-H2)<=1;
if Cond
    ITYPE = 1;
else
    ITYPE = 2;
end
% Create the profile
prof = [] ;
%       (1) / 'TROPICAL
%       (2) / 'MIDLATITUDE SUMMER
%       (3) / 'MIDLATITUDE WINTER corr wala
%       (4) / 'SUBARCTIC SUMMER
%       (5) / 'SUBARCTIC WINTER
%       (6) / 'U. S. STANDARD,  1976

%       (1) / 'TROPICAL 		H20 = 11.02 %	CO2: 0.14 %	02: 88.84 %
%       (2) / 'MIDLATITUDE SUMMER	H20 =  8.22 %	CO2: 0.14 %	02: 91.63 %
%       (3) / 'MIDLATITUDE WINTER	H20 =  2.02 %	CO2: 0.15 %	02: 97.82 %
%       (4) / 'SUBARCTIC SUMMER		H20 =  5.40 %	CO2: 0.15 %	02: 94.45 %
%       (5) / 'SUBARCTIC WINTER		H20 =  0.67 %	CO2: 0.16 %	02: 99.18 %
%       (6) / 'U. S. STANDARD,  1976 	H20 =  0.78 %	CO2: 0.03 %	02: 20.94 %	N2: 78.25 %

prof.model = Season;                      % model number
prof.v1 = 25 ;                       % starting wavenumber
prof.v2 = 334 ;                      % ending wavenumber
prof.h1 = H1;
prof.itype = ITYPE;
dataOut = []; % For defining the output
dataOut.h1 = prof.h1;
switch ITYPE
    case 2
        % For Slant Ranges; ITYPE = 2
        % H1,    H2,   ANGLE,   RANGE,   BETA,   LEN,     HOBS
        prof.rangef_km = RANGE_km;
        prof.h2 = H2;
        
        % prof.zangle = ZANGLE;
        % Output Data
        % dataOut.h2 = prof.h2;
        % dataOut.zangle = prof.zangle;
        dataOut.SlantRangef_km = RANGE_km;%(prof.h2-prof.h1)/cosd(prof.zangle);
%         dataOut.HorizontalRangef_km = dataOut.SlantRangef_km*...
%             sind(prof.zangle);
        %         prof.rangef_km = dataOut.SlantRangef_km;
    case 1
        % For Horizontal Ranges; ITYPE = 1
        % H1,      ,      ,  RANGEF
        prof.h2 = H2;
        prof.h1 = H1;
        prof.rangef_km = RANGE_km;
        % Output Data
        dataOut.h2 = prof.h1;
        % dataOut.zangle = 90;% in degrees
        dataOut.SlantRangef_km = RANGE_km;%prof.rangef_km;
        dataOut.HorizontalRangef_km = prof.rangef_km;
    otherwise
        cd(CodePath)
        error('Choose between 1 and 2')
end
% Run LBLRTM
% .. Run LBLRTM
profout = wrtTape5_prof_Orig(prof);           % Write the TAPE5 file
try
    delete([DirLbl 'TAPE12']);
catch
    disp('File not found...')
    disp('Continuing ...')
end
[output,txt] = system(runlblrtm);                % Run LBLRTM
data_model4 = lbl_read([DirLbl 'TAPE12'], 0, 0);  % Read in result
if VerboseFlag == 1
    disp(txt)
end
dataOut.v = data_model4.v;
dataOut.tau = data_model4.tau;
% % % %      Plot results   % % % % % % % % % %
% .. Plot the radiance
if DisplayFlag == 1
    plot(data_model4.v, data_model4.tau);
    legend('Model 6')
    xlabel('wavenumber')
    ylabel('transmittance')
end

cd(CodePath)

end
%% Flat Band Identifier
function [No] = FlatRegionIdentifier(N,FlatnessCriteria,minNumberOfPoints)
%% [No] = FlatRegionIdentifier(N,FlatnessCriteria,NumberOfPoints,DisplayFlag)
% Input:
% N: Input Signal
% FlatnessCriteria : VErtical threshold
% NumberOfPoints   : minimum number of consective points to be considered
% DisplayFlag      : Plot command for plotting
% Output:
% No               : Indeces of flat regions
%
%
% Example
% For path loss
% FlatnessCriteria = 0.5;% dBm scale
% NumberOfPoints = 10;
% [Idx_Total_PL_Flat] = FlatRegionIdentifier(Atotal_dB(Atotal_dB<=200),FlatnessCriteria,NumberOfPoints,DisplayFlag);
% Example
% For noise
% [Idx_NoiseFlat] = FlatRegionIdentifier(pow2db(Pnoise*1e3),FlatnessCriteria,NumberOfPoints,DisplayFlag);

%% Identify indeces corresponding to Minimum and maximum number of points criteria
FullBandFlag = false; % Assign 0s
Threshold = FlatnessCriteria; % dB scale total variation.
Na = zeros(length(N),'uint16'); % Initial Na with 0s up to the length of N
for ii = 1:length(N) % for loop for the entire length of N
    %     disp(ii)
    Na(ii,ii) = true; % generate a matrix (length N x length N) and assign 1s diagonal of the matrix
    Nb1 = (abs(N(ii)-N)<=Threshold/2); % all entries from ii before and after ii of the matrix which are within the threshold
    IdxNb1 = find(Nb1==false); % find if there are entries which are 0s (i.e. out of the threshold)
    if ~isempty(IdxNb1) % if there are 0s entries
        IdxNb1_ii_less = IdxNb1(find(IdxNb1<ii));% find all entries less than considered point that are out of threshold
        IdxNb1_ii_greater = IdxNb1(find(IdxNb1>ii));% find all entries greater than considered point that are out of threshold
        IdxNb1_ii_less = sort(IdxNb1_ii_less,'descend');
        IdxNb1_ii_greater = sort(IdxNb1_ii_greater,'ascend');
        
        if ii>1 && ~isempty(IdxNb1_ii_less)
            Nb1(1:IdxNb1_ii_less(1)) = false; % Less index entries that are not connecting to consdered point and are in threshold are set to zero
        end
        if ii<length(N) && ~isempty(IdxNb1_ii_greater)
            Nb1(IdxNb1_ii_greater(1):end) = false;% Greater index entries that are not connecting to consdered point and are in threshold are set to zero
        end
    else
        if ii == 1 % first entry (iteration)
            FullBandFlag = true; % FullBandFlag = 1
            break;
        else
            FullBandFlag = false; % FullBandFlag = 1
        end
    end
    Na(ii,:) = Nb1;
end
if ~FullBandFlag % If FullBandFlag is False (i.e. 0)
    N2 = sum(Na,2);
    No = find(N2>=minNumberOfPoints);
else
    No = 1:length(N); % If FullBandFlag is True (i.e. 1), which means the entire band is flat, return the index of the center point of the band
end
end
%% Flat Band Capacity Calculator
% function [Idx_Total_PLpNoise_Flat] = FlatBandCapacityCalculator(Distance,AbsLoss_dB,FreqTHz,ReqIdx)
function [Idx_Total_PLpNoise_Flat] = FlatBandCapacityCalculator(Distance,AbsLoss_dB,FreqTHz,ReqIdx,PLtotal_dB)

deltaFreq = DeltaFreq(FreqTHz*1e12);
NoisePwrReq = NoisePower(AbsLoss_dB,deltaFreq);

% Path Loss Flat Regions
% AbsorptionLossInvalidIdx = find((AbsLoss_dB)>=(max(AbsLoss_dB)-(max(AbsLoss_dB)-min(AbsLoss_dB))/20));
% FlatnessCriteriaPL = Distance*0.01;%+alpha0;
FlatnessCriteriaPL = Distance*0.01+1;

% Distance in meters
% FlatnessCriteriaPL = 50;
% FlatnessCriteriaPL = 1;%0.125; %% dB scale total variation.
NumberOfPoints = 10;    %% -- Bandwidth points
[Idx_Total_PL_Flat_All] = FlatRegionIdentifier(PLtotal_dB,FlatnessCriteriaPL,NumberOfPoints);
% Noise Flat Regions
%% Akhtar
AbsorptionLossThreshold = 400;
AbsorptionLossInvalidIdx = find(AbsLoss_dB>=AbsorptionLossThreshold);
%%
C = intersect(Idx_Total_PL_Flat_All,AbsorptionLossInvalidIdx);
Idx_Total_PL_Flat = setdiff(Idx_Total_PL_Flat_All,C);

FlatnessCriteriaNoise = FlatnessCriteriaPL;%0.125; %% dBm scale total variation.
% NumberOfPoints = 20;    %% --
[Idx_NoiseFlat] = FlatRegionIdentifier(pow2db(NoisePwrReq*1e3),FlatnessCriteriaNoise,NumberOfPoints);
Idx_Total_PLpNoise_Flat = intersect(Idx_Total_PL_Flat(:),Idx_NoiseFlat(:));
Idx_Total_PLpNoise_Flat = intersect(ReqIdx(:),Idx_Total_PLpNoise_Flat(:));
end
%% WaterFilling Power allocation
function [PowerNBProdNsub_dBW,maskOut] = WaterFillingAlgo(maskIn,AbsorptionLoss_dB,PLtotal_dB,TotalTxPower_dBW,Freq_THz)
Idx = find(maskIn==1);
DeltaFreqHz = DeltaFreq(Freq_THz*1e12);

ReqDeltaFreqHz = DeltaFreqHz(Idx);
ReqAbsorptionLoss_dB = AbsorptionLoss_dB(Idx);

PnoiseReqFreq = NoisePower(ReqAbsorptionLoss_dB,ReqDeltaFreqHz);

AtotalReqFreq = db2pow(PLtotal_dB(Idx));
P_Total = db2pow(TotalTxPower_dBW);

errK = 1;
AN_ReqFreq = PnoiseReqFreq.*AtotalReqFreq;
AN_ReqFreq_dummy = AN_ReqFreq;
AN_ReqFreq_dummy(AN_ReqFreq_dummy==0) = [];
AN_ReqFreq(AN_ReqFreq==0) = min(AN_ReqFreq_dummy)*1e-2;
% a = P_Total./sum(1./AN_ReqFreq);
% Sopt = a./AN_ReqFreq;


% For initialization
Kold = db2pow(mean(pow2db(AN_ReqFreq)));% Linear Scale

% fa = find(AN_ReqFreq<Kold);
fa = 1:length(AN_ReqFreq);

if Kold == Inf
    disp('Inf geldi');
end
while errK > 1e-5
    n = length(fa);
    K = (P_Total/n) + (1/n*sum(AN_ReqFreq(fa),'all'));
    fa = find(AN_ReqFreq<K);
    errK = abs(Kold-K)/Kold;
    Kold = K;
    if Kold == Inf
        disp('Inf geldi');
    end
end
Sopt = zeros(size(AN_ReqFreq));
Sopt(fa) = K-AN_ReqFreq(fa);
SoptRegProdNsub = Sopt/sum(Sopt)*P_Total*length(fa);
SoptReg = Sopt/sum(Sopt)*P_Total;

maskOut = zeros(size(maskIn),'like',maskIn);
maskOut(Idx(fa)) = 1;

PowerNBProdNsub_dBW = zeros(size(maskIn));
PowerNBProdNsub_dBW(Idx) = (SoptRegProdNsub);
PowerNBProdNsub_dBW = pow2db(PowerNBProdNsub_dBW);

% PowerNB_dBW = zeros(size(maskIn));
% PowerNB_dBW(Idx) = (SoptReg);
% PowerNB_dBW(Idx) = pow2db(PowerNB_dBW);
end
%% Generate Hemisphere coordinates
function XYZ = GenHemSphCoord(r,theta_d,phi_d,XYZ_Tx,XYZ_Rx)
% r = 20;%meters
% theta_d = 90;%[0:10:180];
% phi_d = [0:10:180];%[90];
% XYZ_Tx = [50,50,100];
[ThetaRxTx_deg,PhiRxTx_deg,d] = Coord2ThetaPhi(XYZ_Tx,XYZ_Rx);
XYZ = [];
for pp = 1:size(XYZ_Tx,1)
    for ii = 1:length(r)
        for jj = 1:length(phi_d)
            for kk = 1:length(theta_d)
                x = r(ii).*cosd(theta_d(kk))*sind(phi_d(jj))+XYZ_Tx(pp,1);
                y = r(ii).*sind(theta_d(kk))*sind(phi_d(jj))+XYZ_Tx(pp,2);
                z = r(ii).*cosd(phi_d(jj))+XYZ_Tx(pp,3);
                XYZ = [XYZ;[XYZ_Tx(pp,1),XYZ_Tx(pp,2),XYZ_Tx(pp,3),x,y,z,theta_d(kk),phi_d(jj)]];
            end
        end
    end
end
end