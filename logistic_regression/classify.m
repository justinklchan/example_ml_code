clear

chirpLen=7200;
template=load('input.txt');
template=template(1:chirpLen);
n=48000;

pksSearchStart=500;
pksSearchEnd=2000;
adist=500;
windowSize=300;

weights=load('weights.txt');
intercept=-2.6863713;
thresh=0.335090254585;

files=["pos","neg"];

figure
hold on
for file = files
    sig = load(strcat(file,".txt"));

    % find 1st sample in each chirp
    % xcorr(cross correlation) ... corr received with transmitted signal.
    % gets ACOR, contains 2 points that are the beginnings.

    [acor,lag]=xcorr(template,sig); % implemented in native C in Android app
    
    % finds the 1-100th largest cross-correlation values to set the minimum
    % peak height parameter in findpeaks
    s=sort(acor,'descend');
    s=s(1:100);

    % find the peaks, but only the peaks that are greater than s(end)-1
    % Question: why s(end)-1, and not s(end).
    % 0.1e5 is the minimum distance between the peaks to find.
    % ref: matlab help findpeaks(select, Fs, 'MinPeakDistance', 0.005)
    [pks,locs]=findpeaks(acor,'MinPeakHeight',s(end)-1,'MinPeakDistance',0.1e5);

    ends = -lag(locs);
    ends=fliplr(ends);

    [~,ind]=max(pks);
    c=sig(ends(ind):ends(ind)+chirpLen);
    p2=abs(fft(c,n));
    p1=p2(1:n/2); 
    chirp=p1(1800:4400-1);

    chirp=movmean(chirp,windowSize);
    dat=(chirp/mean(chirp));

    [pks,locs,widths,proms]=findpeaks(-dat(pksSearchStart:pksSearchEnd-1),'NPeaks',100);
    pks=-pks;

    metric=proms;

    [~,mind]=max(metric);
    maxloc=locs(mind);
    maxpk=pks(mind);
    maxloc=maxloc+pksSearchStart;

    range=dat(maxloc-adist:maxloc+adist-1);
    plot(range);

    logit=sum(range.*weights)+intercept;
    prob=1/(1+exp(-logit));

    if prob > thresh
        ["positive" prob]
    else
        ["negative" prob]
    end
end




























