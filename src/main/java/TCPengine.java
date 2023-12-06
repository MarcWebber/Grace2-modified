import java.util.Iterator;
import java.util.List;
import java.util.Set;

import org.dom4j.Document;
import org.dom4j.DocumentException;
import org.dom4j.Element;
import org.dom4j.io.SAXReader;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.TypeReference;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

import com.google.gson.Gson;

public class TCPengine {
	Document doc = null; 
	String output = null;
	List<Integer> S = new ArrayList<Integer>(); //test sequence after OCP 
	HashMap<Integer, Integer> Med = new HashMap<Integer, Integer>();
	HashMap<Integer, Integer> M_C = new HashMap<Integer, Integer>();
	HashMap<String, Integer> Te = new HashMap<String, Integer>();
	HashMap<Integer, Float> Te_time = new HashMap<Integer, Float>(); // test_time
	HashMap<Integer, Integer> Code = new HashMap<Integer, Integer>();
	HashMap<Integer, Integer> Cla_m = new HashMap<Integer, Integer>();
	HashMap<Integer, Integer> Cla_c = new HashMap<Integer, Integer>();
	List<List<Integer>> M = new ArrayList<List<Integer>>(); //relation matrix
	int Med_num = 0;
	int Code_num = 0;
	int Te_num = 0;
	String package_name = null;
	HashMap<String, Integer> classes = new HashMap<String, Integer>();
	
	
	public void mc_extract(String path) throws DocumentException {
		HashMap<Integer, Integer> M_C_temp = new HashMap<Integer, Integer>();
		HashMap<Integer, Integer> Code_temp = new HashMap<Integer, Integer>();
		int Med_count = 0;
		int Code_count = 0;
		int Te_count = 0;
		int Cla_count = 0;
		SAXReader saxreader =  new SAXReader();
		doc = saxreader.read(path);
		Element root = doc.getRootElement();
		Iterator r_i = root.elementIterator();
		Element project = (Element)r_i.next();
		Element testproject = (Element)r_i.next();
		// System.out.println(testproject.getName());
		// System.out.println(project.getName());
		Iterator t_p = testproject.elementIterator();
		Element t_pac = null;
		while (t_p.hasNext()) {
			Element t_p_e = (Element) t_p.next();
			if (t_p_e.getName().equals("package")) {
				t_pac = t_p_e;
				break;
			}
		}
		Iterator t_pac_i = t_pac.elementIterator();
		while (t_pac_i.hasNext()) {
			Element t_pac_e = (Element) t_pac_i.next();
			if (t_pac_e.getName().equals("file")) {
				Iterator t_f_i = t_pac_e.elementIterator();
				while (t_f_i.hasNext()) {
					Element t_f_e = (Element) t_f_i.next();
					if (t_f_e.getName().equals("line")) {
						if (t_f_e.attributeValue("type").equals("method")) {
							Te.put(t_f_e.attributeValue("signature"), Te_count);
							Te_time.put(Te_count, Float.valueOf(t_f_e.attributeValue("testduration")));
							Te_count += 1;
						}
					}
				}
			}
		}
		
		Te_num = Te_count;
		
		Iterator p = project.elementIterator();
		Element pac = null;
		while (p.hasNext()) {
			Element p_e = (Element) p.next();
			if (p_e.getName().equals("package")) {
				pac = p_e;
				break;
			}
		}
		// System.out.println(pac.getName());
		package_name = pac.attributeValue("name");
		Iterator pac_i = pac.elementIterator();
		while (pac_i.hasNext()) {
			Element pac_e = (Element) pac_i.next();
			if (pac_e.getName().equals("file")) {
				Iterator f_i = pac_e.elementIterator();
				while (f_i.hasNext()) {
					Element f_e = (Element) f_i.next();
					if (f_e.getName().equals("line")) {
						if (f_e.attributeValue("type").equals("method")) {
							Cla_m.put(Med_count + Te_num, Cla_count-1);
							Med.put(Med_count + Te_num, Integer.valueOf(f_e.attributeValue("num")));
							Med_count += 1;
						}
						else if (f_e.attributeValue("type").equals("stmt")) {
							int code_line = Integer.valueOf(f_e.attributeValue("num"));
							int cur_med = Med_count - 1 + Te_num;
							Cla_c.put(Code_count, Cla_count-1);
							M_C_temp.put(Code_count, cur_med);
							Code_temp.put(Code_count, code_line);
							Code_count += 1;
						}
					}
					else if (f_e.getName().equals("class")) {
						classes.put(f_e.attributeValue("name"), null);
						Cla_count ++;
					}
				}
			}
		}
		
		
		Med_num = Med_count;
		Code_num = Code_count;
		List<Integer> temp_zero = null;
		for (int number = 0; number<Te_num+Med_num+Code_num;number++) {
			temp_zero = new ArrayList<Integer>();
			for (int nu = 0; nu<Te_num+Med_num+Code_num;nu++) {
				temp_zero.add(0);
			}
			M.add(temp_zero);
		}
		Iterator i = M_C_temp.entrySet().iterator();
		while (i.hasNext()) {
			HashMap.Entry<Integer, Integer> entry = (HashMap.Entry<Integer, Integer>) i.next();
			int k = entry.getKey();
			int v = entry.getValue();
			M.get(v).set(k + Te_num + Med_num, 1);
			M.get(k + Te_num + Med_num).set(v, 1);
			M_C.put(k + Te_num + Med_num, v);
		}
		Iterator j = Code_temp.entrySet().iterator();
		while (j.hasNext()) {
			HashMap.Entry<Integer, Integer> entry = (HashMap.Entry<Integer, Integer>) j.next();
			int k = entry.getKey();
			int v = entry.getValue();
			Code.put(k + Te_num + Med_num, v);
		}
	}
	
	
	public void js_extrat(String js_part_path) throws IOException{
		//System.out.print(Cla_m);
		String js_path = package_name.replace('.','/');
		String class_js_path;
		FileReader fileReader;
		BufferedReader reader;
		String tempString;
		String lastString = null;
		List<Integer> temp_list;
		int test_num;
		List<List<Integer>> l;
		Set<Integer> med_set = Med.keySet();
		Set<Integer> code_set = Code.keySet();
		Set<String> class_set = classes.keySet();
		int cur_cla = 0;
		for (String c:class_set) {
			class_js_path = js_part_path + js_path + "/"+c+".js";
			fileReader = new FileReader(class_js_path);
			reader = new BufferedReader(fileReader);
			while ((tempString = reader.readLine()) != null) {
				lastString = tempString;
				}
				fileReader.close();
				int idx1 = lastString.indexOf('=');
				String liststring = lastString.substring(idx1+2);
				l = JSON.parseObject(liststring, new TypeReference<List<List<Integer>>>() {});
				//System.out.println(l.get(9));
				for (Integer med:med_set) {
					if (Cla_m.get(med) != cur_cla) {continue;}
					temp_list = l.get(Med.get(med));
					for (int nu=0; nu<temp_list.size();nu++) {
						test_num = temp_list.get(nu);
						M.get(test_num).set(med, 1);
						M.get(med).set(test_num, 1);
					}
				}
				for (Integer cod:code_set) {
					if (Cla_c.get(cod - Te_num - Med_num) != cur_cla) {continue;}
					temp_list = l.get(Code.get(cod));
					for (int nu=0; nu<temp_list.size();nu++) {
						test_num = temp_list.get(nu);
						M.get(test_num).set(cod, 1);
						M.get(cod).set(test_num, 1);
					}
				}
			cur_cla ++;
		}
		
	}
	
	public void OCP() throws IOException {
		float []Cover_w = new float[Te_num];
		int [][]TMmap = new int[Te_num][Med_num];
		for (int n1=0; n1<Te_num;n1++) {
			for (int n2=0; n2<Med_num; n2++) {
				TMmap[n1][n2] = M.get(n1).get(n2 + Te_num);
			}
		}
		HashMap<Integer, Integer> can = new HashMap<Integer, Integer>();
		int priority = Med_num;
		int []Unitcover = new int [Med_num];
		int tmp_num = 0;
		int i;
		int maximum;
		int max_id;
		for (int n1=0; n1<Med_num; n1++) {
			Unitcover[n1] = 0;
		}
		for (int n1=0; n1<Te_num; n1++) {
			can.put(n1, priority);
		}
		while (can.size()>0) {
			maximum = -1;
			max_id = -1;
			i = 0;
			for (HashMap.Entry<Integer, Integer> tp:can.entrySet()) {
				tmp_num = 0;
				for (int n1=0; n1<Med_num; n1++) {
					if (TMmap[i][n1] == 1 && Unitcover[n1] == 0) {tmp_num += 1;}
				}
				can.replace(tp.getKey(), tmp_num);
				if (maximum < tmp_num) {maximum = tmp_num; max_id = tp.getKey();}
				i += 1;
			}
			Cover_w[max_id] = maximum;
			S.add(max_id);
			can.remove(max_id);
		}
		int s_v = 0;
		for (int n1=0; n1<Te_num; n1++) {
			s_v += Cover_w[n1];
		}
		for (int n1=0; n1<Te_num; n1++) {
			Cover_w[n1] /= s_v;
		}
		Gson gson = new Gson();
		String json_CW = gson.toJson(Cover_w);
		FileWriter writer = new FileWriter("./CW.json");
        writer.write(json_CW);
        writer.close();
	}
	
	public void model_predict() throws IOException, InterruptedException {
		Gson gson = new Gson();
        String json_M = gson.toJson(M);
        float []t = new float[Te_num];
        for (HashMap.Entry<Integer, Float> tt:Te_time.entrySet()) {
        	t[tt.getKey()] = tt.getValue();
        }
        String json_t = gson.toJson(t);
        int []num = new int[3];
        num[0] = Te_num;
        num[1] = Med_num;
        num[2] = Code_num;
        String json_num = gson.toJson(num);
        FileWriter writer1 = new FileWriter("./M.json");
        writer1.write(json_M);
        writer1.close();
        FileWriter writer2 = new FileWriter("./t.json");
        writer2.write(json_t);
        writer2.close();
        FileWriter writer3 = new FileWriter("./num.json");
        writer3.write(json_num);
        writer3.close();
        // set the first argument to your python environment
        String [] arg = new String [] {"D:/Anaconda/envs/py3.9withtorch/python", "./Grace2-modified/Grace2/do_task.py", System.getProperty("user.dir")};
		Process proc;
		proc = Runtime.getRuntime().exec(arg);
		BufferedReader in = new BufferedReader(new InputStreamReader(proc.getInputStream()));
        String line;
        while ((line = in.readLine()) != null) {
            //System.out.println(line);
            output = line;
        }
        in.close();
        int code = proc.waitFor();
        //System.out.println(code);
	}
	
	public static void main(String[] args) throws DocumentException, IOException, InterruptedException{
		TCPengine reader = new TCPengine();

		String filePath = args[0];

		String xml_path = filePath + "/target/site/clover/clover.xml";
		String js_part_path = filePath + "/target/site/clover/";
		// set the xml_path to the path of your clover.xml
//		String xml_path = "D:\\eclipse workplace\\maven_helloworld\\target\\site\\clover\\clover.xml";
		reader.mc_extract(xml_path);
		// set the js_part_path to the path of your clover directory name
//		String js_part_path = "D:/eclipse workplace/maven_helloworld/target/site/clover/";
		reader.js_extrat(js_part_path);

		// two args: xml_path,js_part_path
		reader.OCP();
		reader.model_predict();
		System.out.println(reader.output);
	}
}
