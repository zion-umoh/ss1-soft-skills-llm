import fs from 'node:fs/promises';
import path from 'node:path';
import { Presentation, PresentationFile } from '@oai/artifact-tool';
import { resolvePresentationFont, applyPresentationChartFont, finalizePresentation } from '/Users/zion/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations/container_tools/artifact_tool_utils.mjs';

const root='/Users/zion/Developer/School/ss1-soft-skills-llm';
const build=path.join(root,'tmp/poster-build');
const skill='/Users/zion/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
const font=resolvePresentationFont({fontFamily:'Arial'});
const W=841/25.4*96, H=594/25.4*96;
const p=Presentation.create({slideSize:{width:W,height:H}});
const s=p.slides.add(); s.background.fill='#FFFFFF';
const C={ink:'#172E40',teal:'#087E83',grey:'#556772',light:'#EAF4F4',rule:'#D2DEE3',blue:'#527FAD',muted:'#A5B6C0'};
function txt(name,text,x,y,w,h,size=32,bold=false,color=C.ink){
 const z=s.shapes.add({geometry:'textbox',name,position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 z.text=text;z.text.style={typeface:font,fontSize:size,bold,color,autoFit:'none',wrap:'square',verticalAlignment:'top',insets:{left:0,right:0,top:0,bottom:0}};return z;
}
function line(x,y,w){s.shapes.add({geometry:'line',position:{left:x,top:y,width:w,height:0},fill:'none',line:{fill:C.rule,width:2}});}
function heading(text,x,y,w=935){txt(text,text,x,y,w,60,44,true,C.teal);line(x,y+68,w);}
function node(name,text,x,y,w,h){const z=s.shapes.add({geometry:'rect',name,position:{left:x,top:y,width:w,height:h},fill:C.light,line:{fill:C.teal,width:2}});z.text=text;z.text.style={typeface:font,fontSize:32,color:C.ink,alignment:'center',verticalAlignment:'middle',insets:{left:16,right:16,top:12,bottom:12},autoFit:'none'};return z;}
function connect(a,b,from='bottom',to='top'){s.shapes.connect(a,b,{kind:'straight',fromSide:from,toSide:to,line:{fill:C.teal,width:3},tail:{type:'triangle',width:'med',length:'med'}});}

txt('title','Personality Traits and Emotion Recognition to Predict\nEngineering Students’ Soft Skills using applications of LLMs',90,64,3000,205,74,true);
txt('author','Anameti Umoh     Supervisor: Dr. Sangeeta Sangeeta     MSc project, CSC-44120',94,284,2970,52,32,false,C.grey);
line(90,350,2998);
const L=90,M=1110,R=2130;

heading('Background and aims',L,394);
txt('problem','Soft skills matter in engineering education. This project investigates whether interview language and vocal delivery can support their estimation, using personality as an intermediate step.',L,486,935,160);
txt('research-question','Research question',L,678,935,48,35,true);
txt('question','Can LLM-derived text cues and audio characteristics predict interview ratings and support an exploratory Big Five-to-BESSI skill mapping?',L,735,935,145);
txt('objectives','Objectives',L,912,935,48,35,true);
txt('objective-list','1. Extract structured language and audio features.\n2. Compare predictions with simple baselines and published RecruitView results.\n3. Evaluate a separate personality-to-skill mapping and identify its limits.',L,970,935,210);
txt('research-basis','Research basis',L,1218,935,48,35,true);
txt('literature','RecruitView [1] provides interview personality and performance ratings. BESSI research [2, 3] studies related personality and skill measures. Relatedness supports testing a mapping, not assuming that personality and skills are equivalent.',L,1276,935,205);
txt('data','Data',L,1520,935,48,35,true);
txt('data-detail','RecruitView: 1,998 responses, 331 people.\nPaired Big Five/BESSI data: 313 usable rows.\nNo visual features in the final model.',L,1578,935,125);

heading('Method and evaluation',M,394);
const a=node('interview','RecruitView interviews\nTranscript and audio',M+60,490,810,110);
const b=node('features','Feature extraction\nLLM: 10 text cues, including emotional expression\nAudio processing: 12 numeric measurements',M+60,655,810,155);
const c=node('predictor','Model 1: Ridge regression\nPredict Big Five ratings and speaking skills',M+60,865,810,125);
const d=node('mapping','Model 2: linear regression\nUse Big Five predictions to estimate BESSI skills',M+60,1080,810,125);
connect(a,b);connect(b,c);connect(c,d);
txt('mapping-link','Big Five\npredictions only',M+500,1005,360,75,28,false,C.grey);
txt('separate-data','Model 2 learns from separate paired questionnaire data. Statistical scale adjustment connects the two models. Speaking skills remains a separate Model 1 output.',M+25,1237,890,152,30);
txt('why-method','Why this approach?',M,1420,935,45,35,true);
txt('method-reason','LLMs turn wording into reusable numeric features. Compact regression is interpretable and limits complexity with a small sample.',M,1478,935,110);
txt('checks','Evaluation checks',M,1614,935,45,35,true);
txt('checks-detail','Keep each RecruitView participant in one split.\nChoose the model on validation data. Check stability, response length and uncertainty.',M,1670,935,130,31);

heading('Results and interpretation',R,394);
txt('chart-description','Interview prediction: higher correlation is better',R,486,935,45,31,true);
const chart=s.charts.add('bar',{
 position:{left:R-12,top:544,width:960,height:445},
 categories:['Big Five mean','Speaking skills'],
 series:[{name:'Length only',values:[.4284,.4685],fill:C.muted,valuesFormatCode:'0.000'},
 {name:'Audio only',values:[.4645,.4825],fill:C.blue,valuesFormatCode:'0.000'},
 {name:'LLM + audio',values:[.4518,.4859],fill:C.teal,valuesFormatCode:'0.000'}],
 barOptions:{direction:'column',grouping:'clustered',gapWidth:90},hasLegend:true,
 legend:{position:'bottom',textStyle:{typeface:font,fontSize:29,fill:C.ink}},
 xAxis:{textStyle:{typeface:font,fontSize:31,fill:C.ink},line:{fill:C.rule,width:1}},
 yAxis:{min:0,max:.6,majorUnit:.2,numberFormatCode:'0.0',textStyle:{typeface:font,fontSize:28,fill:C.grey},majorGridlines:{fill:C.rule,width:1}},
 dataLabels:{showValue:true,position:'outEnd',textStyle:{typeface:font,fontSize:28,fill:C.ink}},
 chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF'
});applyPresentationChartFont(chart,{fontFamily:font});
txt('metric-note','Spearman correlation, not percentage accuracy.\nTest: 292 responses from 49 participants.',R,1002,935,91,30,false,C.grey);
txt('published','Published paper comparison [1]',R,1130,935,46,35,true);
txt('paper-values','CRMF: Big Five 0.523; speaking skills 0.595.\nThe paper also uses video and a different split. This is a contextual comparison, not a reproduction.',R,1185,935,138,31);
txt('bessi-result','Separate BESSI mapping',R,1358,935,46,35,true);
txt('mapping-values','Mean R² = 0.504 on 47 test rows. Participant independence is unverified. This is not the accuracy of the full interview-to-skills pipeline.',R,1415,935,137,31);
txt('main-finding','Main finding',R,1585,935,46,35,true);
txt('conclusion','The compact model predicts useful patterns. Extra benefit over response length remains uncertain: 95% intervals for the improvements include zero.',R,1643,935,145,32);

line(90,1840,2998);
txt('progress-heading','Progress and completion plan',90,1870,1250,50,38,true,C.teal);
txt('progress','Completed: data, models, comparisons and final checks.\nPoster deadline: 16 September 2026. Proposed plan at right.\nDissertation writing and review remain; dates to confirm.',90,1935,1290,135,30);
const table=s.tables.add({rows:4,columns:4,left:1480,top:1870,width:1607,height:200,columnWidths:[767,280,280,280],values:[['Poster completion','11-12 Sep','13-14 Sep','15-16 Sep'],['Prepare content and layout','Planned','',''],['Seek feedback and revise','','Planned',''],['Final checks and submit PDF','','','Planned']]});
table.borders.assign({fill:C.rule,width:1});
for(let i=0;i<4;i++)for(let j=0;j<4;j++){let cell=table.getCell(i,j);cell.text.style={typeface:font,fontSize:27,color:C.ink,bold:i===0};cell.fill=i===0?'#EAF4F4':(j>0&&cell.value==='Planned'?'#EFF5F8':'#FFFFFF');}
txt('limits','Limits: no direct BESSI labels in RecruitView; emotion cues lack independent validation; engineering-student generalisation is untested.\nFollow-up test analyses are exploratory. Research use only; do not use these estimates for hiring decisions.',90,2090,2998,76,27,false,C.grey);
txt('references','[1] Gupta et al. (2025), RecruitView, arXiv:2512.00450.   [2] Soto et al. (2022), BESSI, doi:10.1037/pspp0000401.\n[3] Sewell et al. (2022), Survey data of social, emotional, and behavioral skills, doi:10.1016/j.dib.2022.107792.',90,2165,2998,52,22,false,C.grey);
s.speakerNotes.textFrame.setText(`Sources: ${root}/docs/dissertation-handoff.md; ${root}/reports/benchmark/recruitview-final-review.md; ${root}/reports/benchmark/recruitview-paper-comparison.md. Published references: https://arxiv.org/abs/2512.00450 ; https://doi.org/10.1037/pspp0000401 ; https://doi.org/10.1016/j.dib.2022.107792 . Course requirements from the earlier Poster Highlights Summary conversation, not independently rechecked against the recording. A1 landscape single-slide poster. User supplied Anameti Umoh, supervisor Dr. Sangeeta Sangeeta and deadline 16 September. The timeline assumes this is the poster deadline in 2026. Proposed review dates are not confirmed appointments. Dissertation completion dates remain to be confirmed. Main model selected on validation; repeated historical test examination makes follow-up findings exploratory. No direct interview BESSI ground truth. Model 2 uses a row-level split and an assumed distributional scale adjustment. Generated chart workbook contains a literal snapshot of the three saved results series.`);
await fs.writeFile(path.join(build,'preview.png'),new Uint8Array(await (await p.export({slide:s,format:'png',scale:.55})).arrayBuffer()));
await fs.writeFile(path.join(build,'layout.json'),await (await s.export({format:'layout'})).text());
const candidate=path.join(build,'candidate.pptx');await(await PresentationFile.exportPptx(p)).save(candidate);
const result=await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:path.join(root,'outputs/poster/soft-skills-research-poster.pptx'),pythonExecutable:'/Users/zion/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','30276000,21384000','--validate-heading-fit','--require-native-table-slide','1'],explicitTotalSlideCount:1,requiredNativeTableOwnerSlides:[1],requiredNativeChartOwnerSlides:[1],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,receiptPath:path.join(build,'validation.json')});
console.log(result);
