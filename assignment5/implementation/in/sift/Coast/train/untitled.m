% number of images = (number of txt files) / 2
files = dir('*.txt');
N = size(files,1)/2;
% store contents of separate descr files in a cell array
A = cell(1,N);
for n = 1:N
    fname = num2str(n);
    while length(fname) < 3
        fname = strcat('0',fname);
    end
    fname = strcat(fname,'_descr.txt');
    A{n} = dlmread(fname);
end
