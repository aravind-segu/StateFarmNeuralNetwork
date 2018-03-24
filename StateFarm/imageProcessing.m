function imageArray = imageProcessing(filePath)
    more off;
    Files = dir(filePath);
    images = [];
    for i = 1:3
      dirName = Files(i).name;
      if(regexp(dirName, '^\w+'))
          dirPath = strcat(filePath, '/', dirName);
          ImageFiles = dir(dirPath);
          for j = 2001:length(ImageFiles)
              imageName = ImageFiles(j).name;
              imgPath = strcat(dirPath, '/', imageName);
              RGB = imread(imgPath);
              RGBResize = imresize(RGB, [480, 480]);
              BW = rgb2gray(RGBResize);
              images = [images; BW(:)'];
              disp(j)
              end
              MatFilePath = strcat(filePath, '/' , strcat(dirName, int2str(5), '.mat'));
              save(MatFilePath, 'images');
              images = [];
         end
      end
    images;
end
