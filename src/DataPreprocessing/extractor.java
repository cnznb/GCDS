import com.alibaba.fastjson.JSON;
import com.csvreader.CsvWriter;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.nio.charset.Charset;
import java.util.Objects;


public class spliter {
    public static void main(String[] args) {
        File rootFolder = new File("E:\\dataset\\xerces");
        traverseFolders(rootFolder);
    }

    public static void solve(String path) {
        try {
            // 读取 JSON 文件，剔除第一行内容
            BufferedReader reader = new BufferedReader(new FileReader(path + "/log.json"));
            reader.readLine(); // 读取掉第一行内容
            String json = "";
            String line = null;
            while ((line = reader.readLine()) != null) {
                json += line;
            }
            reader.close();

            // 解析 JSON 字符串为自定义对象
            ref_json rj = JSON.parseObject(json, ref_json.class);

            CsvWriter csvWriter = new CsvWriter(path + "/class_range.csv",',', Charset.forName("GBK"));
            csvWriter.writeRecord(new String[]{rj.getLeftSideLocations().get(0).getStartLine(), rj.getLeftSideLocations().get(0).getEndLine()});
            CsvWriter cWriter = new CsvWriter(path + "/extract_range.csv",',', Charset.forName("GBK"));
            for(int i=1;i<rj.getLeftSideLocations().size();i++){
                cWriter.writeRecord(new String[]{rj.getLeftSideLocations().get(i).getStartLine(),rj.getLeftSideLocations().get(i).getEndLine()});
            }
            csvWriter.close();
            cWriter.close();
        } catch (Exception ex) {
            ex.printStackTrace();
        }

    }

    public static void traverseFolders(File folder) {
        if (folder.isDirectory()) {
            for (File file : Objects.requireNonNull(folder.listFiles())) {
                if (file.isDirectory() && !file.getName().startsWith("1")) {
                    // 如果遍历到以不 "10" 开头的文件夹，打印文件夹路径
                    System.out.println(file.getAbsolutePath());
                    // 递归遍历此文件夹
                    traverseFolders(file);
                }
                else if (file.isDirectory()) {
                    solve(file.getAbsolutePath());
                }
            }
        }
    }
}
