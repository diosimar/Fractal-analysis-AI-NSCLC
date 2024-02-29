carpetaPrincipal= 'C:\Users\Usuario\Desktop\UNIR_Master IA\TFM\Desarrollo\Imagenes\manifest-1603198545583\NSCLC-Radiomics'
[imagenesDICOM]=CargaSegmentacionesAuto(carpetaPrincipal);
[boxcount,sizes,DF] = dimensionfractal(imagenesDICOM);
DatasetDF(DF);





function [imagenesDICOM] = CargaSegmentacionesAuto(carpetaPrincipal)
%CargaSegmentacionesAuto Función para cargar las imágenes DICOM que
%interesan directamente desde su carpeta.


% Especifica la carpeta principal que contiene subcarpetas y archivos DICOM
%carpetaPrincipal = 'C:\Users\Usuario\Desktop\UNIR_Master IA\TFM\Desarrollo\Imagenes\manifest-1603198545583\NSCLC-Radiomics';

% Obtiene la lista de todas las carpetas dentro de la carpeta principal
subCarpetasNivel1 = dir(carpetaPrincipal);
subCarpetasNivel1 = subCarpetasNivel1([subCarpetasNivel1.isdir]); % Filtra solo directorios
subCarpetasNivel1 = subCarpetasNivel1(~ismember({subCarpetasNivel1.name}, {'.', '..'})); % Elimina las carpetas . y ..
% Inicializa una celda para almacenar las imágenes DICOM
imagenesDICOM = cell(0);

% Recorre las subcarpetas del primer nivel
for i = 1:length(subCarpetasNivel1)
    carpetaNivel1 = fullfile(carpetaPrincipal, subCarpetasNivel1(i).name);

    % Obtiene la lista de subcarpetas del segundo nivel
    subCarpetasNivel2 = dir(carpetaNivel1);
    subCarpetasNivel2 = subCarpetasNivel2([subCarpetasNivel2.isdir]); % Filtra solo directorios
    subCarpetasNivel2 = subCarpetasNivel2(~ismember({subCarpetasNivel2.name}, {'.', '..'})); % Elimina las carpetas . y ..
    
    
    % Recorre las subcarpetas del segundo nivel
    for j = 1:length(subCarpetasNivel2)
        carpetaNivel2 = fullfile(carpetaNivel1, subCarpetasNivel2(j).name);

        % Obtiene la lista de subcarpetas del tercer nivel
        subCarpetasNivel3 = dir(carpetaNivel2);
        subCarpetasNivel3 = subCarpetasNivel3([subCarpetasNivel3.isdir]); % Filtra solo directorios
        subCarpetasNivel3 = subCarpetasNivel3(~ismember({subCarpetasNivel3.name}, {'.', '..'})); % Elimina las carpetas . y ..
        % Filtra las subcarpetas del segundo nivel que contienen "Segmentation" en el nombre
        subCarpetasNivel3 = subCarpetasNivel3(~contains({subCarpetasNivel3.name}, 'Segmentation'));
        % Filtra las subcarpetas del segundo nivel que contienen más de un archivo
        subCarpetasNivel3 = subCarpetasNivel3(arrayfun(@(x) numel(dir(fullfile(carpetaNivel2, x.name, '*.dcm')))<2, subCarpetasNivel3));

        

        % Recorre las subcarpetas del tercer nivel
        for k = 1:length(subCarpetasNivel3)
            carpetaNivel3 = fullfile(carpetaNivel2, subCarpetasNivel3(k).name);

            % Obtiene la lista de archivos DICOM en la carpeta actual
            archivosDICOM = dir(fullfile(carpetaNivel3, '*.dcm'));

            % Recorre todos los archivos DICOM en la carpeta actual
            for m = 1:length(archivosDICOM)
                archivoDICOM = fullfile(carpetaNivel3, archivosDICOM(m).name);

                % Lee la imagen DICOM y almacénala en la celda
                imagenDICOM = dicominfo(archivoDICOM);
                imagenesDICOM{end+1} = imagenDICOM;
            end
        end
    end
end


end

function [boxcounts,sizes,DF] = dimensionfractal(imagenesDICOM)
%dimensionfractal Función para el cálculo de la DF de las imágenes DICOM
%Esta funcion necesita como entrada las imágenes DICOM segmentadas y
%devulve el número de cajas de la función boxcount así como su tamaño.
%También devuelve el valor de la dimensión fractal.

 for i = 1:length(imagenesDICOM)
     %Se saltan 3 imágenes que dieron problema en la carga y cálculo de la
     %DF.
     if i== 128
         i=129;
     end
     if i==317
        i=318;
     end
      if i==388
         i=389;
     end
     info=dicominfo(imagenesDICOM{1,i}.Filename);
     %Se obtienen los contornos de la imagen DICOM para posteriormente
     %acceder a las regiones de interés. Una de las regiones es el tumor
     %segmentado GTV-1.
     rtContours=dicomContours(info);
     rtContours.ROIs
     plotContour(rtContours);
     referenceInfo = imref3d([128 128 50],xlim,ylim,zlim);
     %Se cera una máscara sobre el tumor para trabajar a continuación solo
     %con esa zona de interés de la imagen.
     rtMask = createMask(rtContours,'GTV-1',referenceInfo);
     %Se aplicaca la función boxcount y polyfit para obtener la dimensión fractal.
    [boxcountData, size] = boxcount(rtMask);
    Dimensionfractal = polyfit(log(size), log(boxcountData), 1);
    DF{i} = abs(Dimensionfractal(1));
    boxcounts{i} = boxcountData;
    sizes{i} = size;
 end
end
function DatasetDF(DF)
%DatasetDF Función para agregar el dato de dimensión fractal al resto de
%datos del dataset.
%Se traspone el array de la dimensión fractal para que los datos queden en
%una columna.
DF_tras=DF';
%Se asignan valor 999 a los 3 pacientes para los cuales no fue posible
%calcular la dimension fractal.
DF_tras(128)={NaN};
DF_tras(317)={1.2057};
DF_tras(388)={1.0796};

%Se crea el cell de la dimensión fractal ya con la cabecera
DF_Final={'F.analysis'};
for i=1:422

DF_Final{i+1,1}=DF_tras{i,1};

end
%Se escriben los valores de dimensión fractal en un nuevo csv
writecell( DF_Final, 'DF.csv','Delimiter',',');
%Se leen y se concatenan los dos csv. El csv del dataset y el csv de la
%dimensión fractal.
csv1=readcell('NSCLC.csv');
allcsv=horzcat(csv1,DF_Final);
%Se escribe la matriz concatenada en un nuevo csv que será el conjunto de
%datos que se utilice para realizar los modelos.
writecell(allcsv,'NSCLC_DF.csv');
end