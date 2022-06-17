include("../../../config.jl");
using Downloads

url = "https://nextcloud.jyu.fi/index.php/s/Xfre5755j8C5xCB/download";
outputfolder = joinpath(SIMEXP_PATH);
mkpath(outputfolder);
outputfile = joinpath(outputfolder, "simulation-data.zip");
Downloads.download(url, outputfile);

run(`unzip $outputfile -d $outputfolder`);
run(`rm $outputfile`);

println("Simulation data downloaded to $outputfolder.")
