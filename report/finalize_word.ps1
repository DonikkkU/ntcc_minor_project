param(
    [string]$Docx = "$PSScriptRoot\NTCC_PD_Report.docx",
    [string]$Pdf = "$PSScriptRoot\NTCC_PD_Report.pdf"
)
# Opens the report in MS Word, updates the table of contents, the lists of
# tables and figures and all page fields, saves it and exports a PDF copy.
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($Docx)
    $doc.Repaginate()
    foreach ($t in $doc.TablesOfContents) { $t.Update() }
    $doc.Fields.Update() | Out-Null
    foreach ($s in $doc.Sections) {
        foreach ($h in $s.Headers) { $h.Range.Fields.Update() | Out-Null }
        foreach ($f in $s.Footers) { $f.Range.Fields.Update() | Out-Null }
    }
    $doc.Repaginate()
    foreach ($t in $doc.TablesOfContents) { $t.Update() }
    $doc.Save()
    $doc.ExportAsFixedFormat($Pdf, 17)
    $words = $doc.ComputeStatistics(0)
    $pages = $doc.ComputeStatistics(2)
    Write-Output "words=$words pages=$pages"
    $doc.Close()
}
finally {
    $word.Quit()
}
