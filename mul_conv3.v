// 4x4 

module mul_conv3#(
parameter CH_NUM = 4,
parameter ACT_PER_ADDR = 4,
parameter BW_PER_ACT = 12,
parameter WEIGHT_PER_ADDR = 9, 
parameter BIAS_PER_ADDR = 1,
parameter BW_PER_PARAM = 8,
parameter activation_fl = 4,
parameter weight_fl = 7,
parameter bias_fl = 7
)
(
input [WEIGHT_PER_ADDR*BW_PER_PARAM-1:0] weight0,
input [WEIGHT_PER_ADDR*BW_PER_PARAM-1:0] weight1,
input [WEIGHT_PER_ADDR*BW_PER_PARAM-1:0] weight2,
input [WEIGHT_PER_ADDR*BW_PER_PARAM-1:0] weight3,
input [WEIGHT_PER_ADDR*BW_PER_PARAM-1:0] weight4,
input [WEIGHT_PER_ADDR*BW_PER_PARAM-1:0] weight5,
input [WEIGHT_PER_ADDR*BW_PER_PARAM-1:0] weight6,
input [WEIGHT_PER_ADDR*BW_PER_PARAM-1:0] weight7,
input [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] map0,
input [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] map1,
input [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] map2,
input [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] map3,
input [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] map4,
input [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] map5,
input [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] map6,
input [CH_NUM*ACT_PER_ADDR*BW_PER_ACT-1:0] map7,
input signed [BW_PER_PARAM-1:0] bias,
input [1:0] cnt,
input [11:0] residue,
output reg signed [BW_PER_ACT-1:0] out
);    
integer i, j, k;

reg [WEIGHT_PER_ADDR*BW_PER_ACT-1:0] mul_in_ch0; // 3*3*12 = 98
reg [WEIGHT_PER_ADDR*BW_PER_ACT-1:0] mul_in_ch1; 
reg [WEIGHT_PER_ADDR*BW_PER_ACT-1:0] mul_in_ch2;
reg [WEIGHT_PER_ADDR*BW_PER_ACT-1:0] mul_in_ch3;
reg [WEIGHT_PER_ADDR*BW_PER_ACT-1:0] mul_in_ch4; // 3*3*12 = 98
reg [WEIGHT_PER_ADDR*BW_PER_ACT-1:0] mul_in_ch5; 
reg [WEIGHT_PER_ADDR*BW_PER_ACT-1:0] mul_in_ch6;
reg [WEIGHT_PER_ADDR*BW_PER_ACT-1:0] mul_in_ch7;

reg signed [BW_PER_ACT-1:0] ifmap [0:7][0:2][0:2]; // 4ch
reg signed [BW_PER_PARAM-1:0] weight [0:7][0:2][0:2]; // 4ch
reg signed [BW_PER_ACT+BW_PER_PARAM-1:0] psum [0:7][0:2][0:2];
reg signed [BW_PER_ACT+BW_PER_PARAM+7-1:0] psum_mul_all;
reg signed [BW_PER_ACT+BW_PER_PARAM+7-1:0] psum_mul_all_1;
reg signed [BW_PER_ACT+BW_PER_PARAM+7-1:0] psum_bias;
reg signed [BW_PER_PARAM+8-1:0] bias_shift;
reg signed [BW_PER_ACT+BW_PER_PARAM+7-1:0] psum_bias_shift;
reg signed [BW_PER_ACT-1:0] out_prerelu;

reg signed [BW_PER_ACT+BW_PER_PARAM+7-1:0] relu_output;
reg signed [BW_PER_ACT+BW_PER_PARAM+7-1:0] rounding_output;
always@* begin
	for(i = 0; i < 8; i = i + 1) begin
		for(j = 0; j < 3; j = j + 1) begin
			for(k = 0; k < 3; k = k + 1) begin
				psum[i][j][k] = ifmap[i][j][k] * weight[i][j][k];
			end
		end
	end
	psum_mul_all = psum[0][0][0] + psum[0][0][1] + psum[0][0][2] + psum[0][1][0] + psum[0][1][1] + psum[0][1][2] + psum[0][2][0] + psum[0][2][1] + psum[0][2][2] + psum[1][0][0] + psum[1][0][1] + psum[1][0][2] + psum[1][1][0] + psum[1][1][1] + psum[1][1][2] + psum[1][2][0] + psum[1][2][1] + psum[1][2][2] + psum[2][0][0] + psum[2][0][1] + psum[2][0][2] + psum[2][1][0] + psum[2][1][1] + psum[2][1][2] + psum[2][2][0] + psum[2][2][1] + psum[2][2][2] + psum[3][0][0] + psum[3][0][1] + psum[3][0][2] + psum[3][1][0] + psum[3][1][1] + psum[3][1][2] + psum[3][2][0] + psum[3][2][1] + psum[3][2][2]
				 + psum[4][0][0] + psum[4][0][1] + psum[4][0][2] + psum[4][1][0] + psum[4][1][1] + psum[4][1][2] + psum[4][2][0] + psum[4][2][1] + psum[4][2][2] + psum[5][0][0] + psum[5][0][1] + psum[5][0][2] + psum[5][1][0] + psum[5][1][1] + psum[5][1][2] + psum[5][2][0] + psum[5][2][1] + psum[5][2][2] + psum[6][0][0] + psum[6][0][1] + psum[6][0][2] + psum[6][1][0] + psum[6][1][1] + psum[6][1][2] + psum[6][2][0] + psum[6][2][1] + psum[6][2][2] + psum[7][0][0] + psum[7][0][1] + psum[7][0][2] + psum[7][1][0] + psum[7][1][1] + psum[7][1][2] + psum[7][2][0] + psum[7][2][1] + psum[7][2][2];
	psum_mul_all_1 = psum_mul_all +	(residue << 7) ;
	// psum_bias = psum_mul_all + bias_shift + 64;
	psum_bias = psum_mul_all_1 + (bias << activation_fl + weight_fl - bias_fl);
	relu_output = psum_bias < 0 ? 0 : psum_bias;
	rounding_output = relu_output + (1 << weight_fl - 1);
	psum_bias_shift = rounding_output >>> weight_fl;
	if (psum_bias_shift > 2047) out = 2047;
	else if (psum_bias_shift < -2048) out = -2048;
	else out = psum_bias_shift[11:0];
	// psum_bias_shift = psum_bias[BW_PER_ACT+BW_PER_PARAM+6-1:7]; // 12 + 8 + 6 - 1 25:7
	// if(psum_bias_shift > 2047) out_prerelu = 2047;
	// else if(psum_bias_shift < -2048) out_prerelu = -2048;
	// else out_prerelu = psum_bias_shift[11:0];
	// if(out_prerelu < 0) out = 0;
	// else out = out_prerelu;
end

always@* begin
	for(i = 0; i < 3; i = i + 1) begin
		for(j = 0; j < 3; j = j + 1) begin
			ifmap[0][i][j] = mul_in_ch0[(3*i+j+1)*12-1-:12];
			ifmap[1][i][j] = mul_in_ch1[(3*i+j+1)*12-1-:12];
			ifmap[2][i][j] = mul_in_ch2[(3*i+j+1)*12-1-:12];
			ifmap[3][i][j] = mul_in_ch3[(3*i+j+1)*12-1-:12];
			ifmap[4][i][j] = mul_in_ch4[(3*i+j+1)*12-1-:12];
			ifmap[5][i][j] = mul_in_ch5[(3*i+j+1)*12-1-:12];
			ifmap[6][i][j] = mul_in_ch6[(3*i+j+1)*12-1-:12];
			ifmap[7][i][j] = mul_in_ch7[(3*i+j+1)*12-1-:12];
			weight[0][i][j] = weight0[(3*i+j+1)*8-1-:8];
			weight[1][i][j] = weight1[(3*i+j+1)*8-1-:8];
			weight[2][i][j] = weight2[(3*i+j+1)*8-1-:8];
			weight[3][i][j] = weight3[(3*i+j+1)*8-1-:8];
			weight[4][i][j] = weight4[(3*i+j+1)*8-1-:8];
			weight[5][i][j] = weight5[(3*i+j+1)*8-1-:8];
			weight[6][i][j] = weight6[(3*i+j+1)*8-1-:8];
			weight[7][i][j] = weight7[(3*i+j+1)*8-1-:8];
		end
	end	
end

always@* begin
	mul_in_ch0 = 0;
	mul_in_ch1 = 0;
	mul_in_ch2 = 0;
	mul_in_ch3 = 0;
	mul_in_ch4 = 0;
	mul_in_ch5 = 0;
	mul_in_ch6 = 0;
	mul_in_ch7 = 0;
	case(cnt) // 0 for UL 1 for UR 2 FOR LL 3 FOR LR
		0: begin
			mul_in_ch0 = {map0[191:180], map0[179:168], map1[191:180], map0[167:156], map0[155:144], map1[167:156], map2[191:180], map2[179:168], map3[191:180]};
			mul_in_ch1 = {map0[143:132], map0[131:120], map1[143:132], map0[119:108], map0[107:96], map1[119:108], map2[143:132], map2[131:120], map3[143:132]};
			mul_in_ch2 = {map0[95:84], map0[83:72], map1[95:84], map0[71:60], map0[59:48], map1[71:60], map2[95:84], map2[83:72], map3[95:84]};
			mul_in_ch3 = {map0[47:36], map0[35:24], map1[47:36], map0[23:12], map0[11:0], map1[23:12], map2[47:36], map2[35:24], map3[47:36]};
			mul_in_ch4 = {map4[191:180], map4[179:168], map5[191:180], map4[167:156], map4[155:144], map5[167:156], map6[191:180], map6[179:168], map7[191:180]};
			mul_in_ch5 = {map4[143:132], map4[131:120], map5[143:132], map4[119:108], map4[107:96], map5[119:108], map6[143:132], map6[131:120], map7[143:132]};
			mul_in_ch6 = {map4[95:84], map4[83:72], map5[95:84], map4[71:60], map4[59:48], map5[71:60], map6[95:84], map6[83:72], map7[95:84]};
			mul_in_ch7 = {map4[47:36], map4[35:24], map5[47:36], map4[23:12], map4[11:0], map5[23:12], map6[47:36], map6[35:24], map7[47:36]};
		end
		1: begin
			mul_in_ch0 = {map0[179:168], map1[191:180], map1[179:168], map0[155:144], map1[167:156], map1[155:144], map2[179:168], map3[191:180], map3[179:168]};
			mul_in_ch1 = {map0[131:120], map1[143:132], map1[131:120], map0[107:96], map1[119:108], map1[107:96], map2[131:120], map3[143:132], map3[131:120]};
			mul_in_ch2 = {map0[83:72], map1[95:84], map1[83:72], map0[59:48], map1[71:60], map1[59:48], map2[83:72], map3[95:84], map3[83:72]};
			mul_in_ch3 = {map0[35:24], map1[47:36], map1[35:24], map0[11:0], map1[23:12], map1[11:0], map2[35:24], map3[47:36], map3[35:24]};
			mul_in_ch4 = {map4[179:168], map5[191:180], map5[179:168], map4[155:144], map5[167:156], map5[155:144], map6[179:168], map7[191:180], map7[179:168]};
			mul_in_ch5 = {map4[131:120], map5[143:132], map5[131:120], map4[107:96], map5[119:108], map5[107:96], map6[131:120], map7[143:132], map7[131:120]};
			mul_in_ch6 = {map4[83:72], map5[95:84], map5[83:72], map4[59:48], map5[71:60], map5[59:48], map6[83:72], map7[95:84], map7[83:72]};
			mul_in_ch7 = {map4[35:24], map5[47:36], map5[35:24], map4[11:0], map5[23:12], map5[11:0], map6[35:24], map7[47:36], map7[35:24]};
		end
		2: begin
			mul_in_ch0 = {map0[167:156], map0[155:144], map1[167:156], map2[191:180], map2[179:168], map3[191:180], map2[167:156], map2[155:144], map3[167:156]};
			mul_in_ch1 = {map0[119:108], map0[107:96], map1[119:108], map2[143:132], map2[131:120], map3[143:132], map2[119:108], map2[107:96], map3[119:108]};
			mul_in_ch2 = {map0[71:60], map0[59:48], map1[71:60], map2[95:84], map2[83:72], map3[95:84], map2[71:60], map2[59:48], map3[71:60]};
			mul_in_ch3 = {map0[23:12], map0[11:0], map1[23:12], map2[47:36], map2[35:24], map3[47:36], map2[23:12], map2[11:0], map3[23:12]};
			mul_in_ch4 = {map4[167:156], map4[155:144], map5[167:156], map6[191:180], map6[179:168], map7[191:180], map6[167:156], map6[155:144], map7[167:156]};
			mul_in_ch5 = {map4[119:108], map4[107:96], map5[119:108], map6[143:132], map6[131:120], map7[143:132], map6[119:108], map6[107:96], map7[119:108]};
			mul_in_ch6 = {map4[71:60], map4[59:48], map5[71:60], map6[95:84], map6[83:72], map7[95:84], map6[71:60], map6[59:48], map7[71:60]};
			mul_in_ch7 = {map4[23:12], map4[11:0], map5[23:12], map6[47:36], map6[35:24], map7[47:36], map6[23:12], map6[11:0], map7[23:12]};
		end		
		3: begin
			mul_in_ch0 = {map0[155:144], map1[167:156], map1[155:144], map2[179:168], map3[191:180], map3[179:168], map2[155:144], map3[167:156], map3[155:144]};
			mul_in_ch1 = {map0[107:96], map1[119:108], map1[107:96], map2[131:120], map3[143:132], map3[131:120], map2[107:96], map3[119:108], map3[107:96]};
			mul_in_ch2 = {map0[59:48], map1[71:60], map1[59:48], map2[83:72], map3[95:84], map3[83:72], map2[59:48], map3[71:60], map3[59:48]};
			mul_in_ch3 = {map0[11:0], map1[23:12], map1[11:0], map2[35:24], map3[47:36], map3[35:24], map2[11:0], map3[23:12], map3[11:0]};
			mul_in_ch4 = {map4[155:144], map5[167:156], map5[155:144], map6[179:168], map7[191:180], map7[179:168], map6[155:144], map7[167:156], map7[155:144]};
			mul_in_ch5 = {map4[107:96], map5[119:108], map5[107:96], map6[131:120], map7[143:132], map7[131:120], map6[107:96], map7[119:108], map7[107:96]};
			mul_in_ch6 = {map4[59:48], map5[71:60], map5[59:48], map6[83:72], map7[95:84], map7[83:72], map6[59:48], map7[71:60], map7[59:48]};
			mul_in_ch7 = {map4[11:0], map5[23:12], map5[11:0], map6[35:24], map7[47:36], map7[35:24], map6[11:0], map7[23:12], map7[11:0]};
		end
		default: begin
			mul_in_ch0 = 0;
			mul_in_ch1 = 0;
			mul_in_ch2 = 0;
			mul_in_ch3 = 0;
			mul_in_ch4 = 0;
			mul_in_ch5 = 0;
			mul_in_ch6 = 0;
			mul_in_ch7 = 0;
		end
	endcase
end
endmodule